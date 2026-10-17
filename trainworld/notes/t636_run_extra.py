import t636_run_all as R
from concurrent.futures import ThreadPoolExecutor
S = R.SIMPLE
R.SETS.update({
    "w15": S + ",asc_walk=1.5",
    "b100w15": S + ",beta=0.1,asc_walk=1.5",
    "d3aw15": S + ",beta=0.075,damp_min=26.67,asc_walk=1.5",
})
jobs = [(k, c, n) for k in ("test", "calib") for c in R.CITIES for n in ("w15", "b100w15", "d3aw15")]
with ThreadPoolExecutor(6) as ex:
    list(ex.map(R.job, jobs))
