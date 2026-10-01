"""python dump_state.py <cc> : run build_model.main's steps up to drop_unridden_sections
(read only: writes nothing in the project) and pickle (lines, stations, route_share) to
the scratch folder, so gtfs_served can be iterated on without a full build."""
import os
import pickle
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, r"C:\Users\anita\projects\maps\noritetsu")
os.chdir(r"C:\Users\anita\projects\maps\noritetsu")
import build_model as bm  # noqa: E402

cc = sys.argv[1]
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)


import importlib  # noqa: E402
lines, stations, geoms, way_lines = bm.build(cc, log)
for l in lines:
    l.setdefault("src", "osm")
mod = importlib.import_module("rinf")
lines, stations, geoms = bm.merge_sources((lines, stations, geoms),
                                          mod.build(str(bm.ROOT / "data/raw/rinf" / cc), log), log)
import line_colours  # noqa: E402
line_colours.apply(cc, lines, log)
reg_ways, route_share = bm.register_way_lines(cc, lines, geoms, log)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"state_{cc}.pkl")
with open(out, "wb") as f:
    pickle.dump((lines, stations, geoms, route_share), f)
log(f"wrote {out}")
