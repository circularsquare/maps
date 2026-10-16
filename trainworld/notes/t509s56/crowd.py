"""Seoul at crowd_alpha 0.5, 1, 3, 4: Line 9's express share of person-stops and the score."""
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

TW = r"C:\Users\anita\projects\maps\trainworld"
PY = r"C:\Users\anita\projects\maps\venv\Scripts\python.exe"
env = dict(os.environ, PYTHONIOENCODING="utf-8")


def go(a):
    tag = f"crowd{a:g}"
    subprocess.run([PY, os.path.join(TW, "pipeline", "calib.py"), "run", "seo", "--tag", tag, "--set", f"crowd_alpha={a:g}"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    return a, tag


with ThreadPoolExecutor(4) as ex:
    done = list(ex.map(go, [0.5, 1.0, 3.0, 4.0]))
for a, tag in [(2.0, "s5base")] + done:
    d = os.path.join(TW, "data", "work", "calib", "seo")
    r = json.load(open(os.path.join(d, f"run_{tag}.json")))
    s = json.load(open(os.path.join(d, f"score_{tag}.json")))
    ld = r["lines"]["load"]
    print(f"alpha {a:g}: line 9 express share {ld[20] / (ld[19] + ld[20]):.1%} (published 40.2%); score {s['total']:.3f} "
          f"stations {s['parts']['stations']:.3f} level x{2.71828 ** s['station_level']:.2f}")
