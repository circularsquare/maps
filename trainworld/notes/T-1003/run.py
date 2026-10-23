"""T-1003: the walk harness for each setting and pack, 2 at a time; skips outputs that exist.

    python run.py <setting,..> [packs]
"""
import concurrent.futures as cf
import glob
import os
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
TW = "C:/Users/anita/projects/maps/trainworld"
EXE = sorted(glob.glob(f"{D}/copy/sim/target/release/deps/trainworld_sim-*.exe"), key=os.path.getmtime)[-1]
SETS = {
    "base": "",
    "nobonus": "walk_bonus_min=0",
    "bonus5": "walk_bonus_min=5",
    "nohome": "home_park_f=0",
    "nopaid": "park_paid_min=0",
    "cap20": "park_paid_cap_min=20",
    "cap15home": "park_paid_cap_min=15",
    "city25": "road_city_kmh=25",
    "walk2": "walk_trip_w=2",
    "walk1": "walk_trip_w=1",
    "home02": "home_park_f=0.2",
    "slope10": "park_paid_min=10",
}
CN = "sha,bei,gua,shz,cho,chg,wuh,xia,tia,hag,zib,tay,gui,nag,tag,nat,jii,tik,huz,zhb,fug,hoe,sio,zii"
OTHER = "nyc,lon,par,tyo,mun,lyo,lee,mil,seo,mum,hon,tai"
OUT = f"{D}/out"
os.makedirs(OUT, exist_ok=True)
names = sys.argv[1].split(",")
packs = sys.argv[2].split(",") if len(sys.argv) > 2 else (CN + "," + OTHER).split(",")
if len(sys.argv) > 2 and sys.argv[2].startswith("@"):
    packs = open(f"{D}/{sys.argv[2][1:]}").read().strip().split(",")


def one(c, v):
    out = f"{OUT}/{c}_{v}.json"
    if os.path.exists(out):
        return c, v, "cached"
    env = dict(os.environ, T1003_PACK=f"{TW}/data/packs/{c}.json", T1003_OUT=out, T1003_SET=SETS.get(v, ""), RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")
    if v.startswith("oth"):  # oth10_5: the other way at 10 + 5 a km
        env["T1003_OTHER"] = v[3:].replace("_", ",")
    t = time.time()
    r = subprocess.run([EXE, "t1003::t1003", "--ignored", "--test-threads", "1"], env=env, capture_output=True, text=True)
    return c, v, f"{time.time() - t:.0f}s rc {r.returncode} {r.stdout.strip()[-400:] if r.returncode else ''}"


with cf.ThreadPoolExecutor(2) as ex:
    for res in ex.map(lambda cv: one(*cv), [(c, v) for v in names for c in packs]):
        print(*res, flush=True)
