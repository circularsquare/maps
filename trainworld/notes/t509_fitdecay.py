"""The game's mean outing length by purpose against the decay length, per pack (calib --info-only).

    python notes/t509_fitdecay.py <pack id> [<pack json>]   -> artifacts/trainworld/t509s4/<id>.json, printed
(the pack's own `everyday` decays are overridden; T-509 stage 4)
"""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

TW = r"C:\Users\anita\projects\maps\trainworld"
BIN = os.path.join(TW, "data", "work", "target", "release", "calib.exe")
NET = os.path.join(TW, "data", "work", "realnet", "sdj_real.json")  # read, unused with --info-only
HERE = os.path.join(TW, "..", "artifacts", "trainworld", "t509s4")
DECAYS = [1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 8000]
KINDS = ["errands", "social", "school", "university", "airport", "medical"]

city = sys.argv[1]
pack = sys.argv[2] if len(sys.argv) > 2 else os.path.join(TW, "data", "packs", f"{city}.json")
os.makedirs(HERE, exist_ok=True)


def one(dm):
    out = os.path.join(HERE,f"{city}_{dm}.json")
    s = ",".join(f"decay_{k}={dm}" for k in ("errands", "social", "medical"))
    subprocess.run([BIN, "--net", NET, "--pack", pack, "--out", out, "--info-only", "--set", s], check=True,
                   stderr=subprocess.DEVNULL)
    info = json.load(open(out))["everyday_info"]
    os.remove(out)
    return dm, {KINDS[i]: info[3 * i + 2] for i in range(len(info) // 3)}


with ThreadPoolExecutor(int(os.environ.get("JOBS", "3"))) as ex:
    res = dict(ex.map(one, DECAYS))
json.dump({"pack": pack, "decays": DECAYS, "km": {str(k): v for k, v in res.items()}},
          open(os.path.join(HERE,f"{city}.json"), "w"), indent=1)
for dm in DECAYS:
    k = res[dm]
    print(f"{city} decay {dm:5d}: errands {k['errands']:.2f}  social {k['social']:.2f}  medical {k.get('medical', 0):.2f}")
