import t636_run_all as R
from concurrent.futures import ThreadPoolExecutor
S = R.SIMPLE
NEW = S + ",beta=0.1,asc_walk=1.0,walk_trip_w_far=1.5,park_min=5"
R.SETS.update({"new": NEW, "newp3": S + ",beta=0.1,asc_walk=1.0,walk_trip_w_far=1.5"})
jobs = [("test", c, n) for c in R.CITIES for n in ("today", "simple", "b100", "new", "newp3")] + [("calib", c, n) for c in R.CITIES for n in ("new", "newp3")]
with ThreadPoolExecutor(6) as ex:
    list(ex.map(R.job, jobs))
