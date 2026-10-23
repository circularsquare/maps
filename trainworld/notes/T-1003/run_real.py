import os, sys
sys.argv = [sys.argv[0], "none"]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_calib import run
SETS = {"base": "", "home02": "home_park_f=0.2", "nohome": "home_park_f=0", "slope10": "park_paid_min=10",
        "nobonus": "walk_bonus_min=0", "cap20": "park_paid_cap_min=20"}
RN = "C:/Users/anita/projects/maps/trainworld/data/work/realnet"
for city in ("nyc", "lon"):
    for v, s in SETS.items():
        print(*run(f"real_{city}_{v}", f"{RN}/{city}_real.json", city, s), flush=True)
