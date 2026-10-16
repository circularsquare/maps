"""T-460: Lagos rebuilt with no cap keys in its config (the defaults apply) into a scratch folder,
against the pack inputs it has now. Never touches data/packs or data/work.

    ..\\venv\\Scripts\\python.exe -u trainworld/notes/t460_lagos_check.py <scratch folder>

Needs Nigeria's cached features (osm_features.py africa/nigeria). Compares: the OSM job-place rows
(the four columns the old cache has), the tier 0 signals arrays, and the cells.bin the build writes.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import filecmp
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as b  # noqa: E402
import read_pack  # noqa: E402
import tier0  # noqa: E402

SCRATCH = os.path.abspath(sys.argv[1])
REAL_WORK, REAL_RAW = b.WORK, tier0.raw("osm_jobs_lag.csv.gz")
b.WORK = os.path.join(SCRATCH, "work")
b.PACKS = os.path.join(SCRATCH, "packs")
read_pack.load_pack.__defaults__ = (b.PACKS,)
orig_raw = tier0.raw
tier0.raw = lambda *p: os.path.join(SCRATCH, p[0]) if p and p[0] == "osm_jobs_lag.csv.gz" else orig_raw(*p)
tier0.finish = lambda *a, **k: None
assert "land_cap_km2" not in b.CITIES["lag"] and "land_spread_deg" not in b.CITIES["lag"]

os.makedirs(b.WORK, exist_ok=True)
tier0.main("lag")

old, new = pd.read_csv(REAL_RAW), pd.read_csv(os.path.join(SCRATCH, "osm_jobs_lag.csv.gz"))
rows_same = old.equals(new[list(old.columns)])
print(f"OSM rows: {len(old):,} old, {len(new):,} new, the four columns equal: {rows_same}")
a, z = np.load(os.path.join(REAL_WORK, "lag.tier0_signals.npz")), np.load(os.path.join(b.WORK, "lag.tier0_signals.npz"))
sig_same = sorted(a.files) == sorted(z.files) and all(np.array_equal(a[k], z[k]) for k in a.files)
print("signals arrays identical:", sig_same, {k: bool(np.array_equal(a[k], z[k])) for k in a.files})
bin_same = filecmp.cmp(os.path.join(REAL_WORK, "lag.cells.bin"), os.path.join(b.WORK, "lag.cells.bin"), shallow=False)
print("cells.bin identical:", bin_same)
print("ALL IDENTICAL" if rows_same and sig_same and bin_same else "DIFFERENT")
