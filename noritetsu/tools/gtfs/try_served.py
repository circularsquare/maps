"""python try_served.py <cc> : the hook's steps on the pickled state from dump_state.py, with
and without gtfs_served, writing nothing in the project but gtfs_served's own cache."""
import copy
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, r"C:\Users\anita\projects\maps\noritetsu")
sys.stdout.reconfigure(encoding="utf-8")
import build_model as bm  # noqa: E402
import gtfs_served  # noqa: E402

gtfs_served.WRITE_CACHE = False     # inspecting data only reads it

cc = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, f"state_{cc}.pkl"), "rb") as f:
    lines, stations, geoms, route_share = pickle.load(f)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


def quiet(m):
    pass


L0, S0, G0, R0 = copy.deepcopy((lines, stations, geoms, route_share))
bm.drop_unridden_sections(L0, S0, G0, R0, log)
base = {(l["id"], f"{s[0]}|{s[1]}"): s[2] for l in L0 if l["src"] != "osm" for s in l["sections"]}

res = gtfs_served.check(cc, lines, stations, route_share, log)
drop = bm.drop_unridden_sections(lines, stations, geoms, route_share, log)
lines = [l for l in lines if l["id"] not in drop]
gtfs_served.mark(res, lines, log)
now = {(l["id"], f"{s[0]}|{s[1]}"): s[2] for l in lines if l["src"] != "osm" for s in l["sections"]}
closed = {(l["id"], k) for l in lines for k in l.get("closed", [])}
print(f"register km without the feed {sum(base.values()):,.0f}, with {sum(now.values()):,.0f}; "
      f"added {sum(v for k, v in now.items() if k not in base):,.0f}, "
      f"removed {sum(v for k, v in base.items() if k not in now):,.0f}; "
      f"closed {sum(now[k] for k in closed if k in now):,.0f} km")
