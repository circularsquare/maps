"""T-1150 rebuild, step 1: the GHS-BUILT-V total (E2020) tiles into data/raw/tier0/ghsl/, where
tier0._tile_path looks: the nine the tests fetched move from data/raw/t1150_ghsl/, the rest download
through tier0 itself (about 15-50 MB each).

    python trainworld/notes/t1150/rebuild_tiles.py            the China metros' ten tiles
    python trainworld/notes/t1150/rebuild_tiles.py R4_C26 ...  these tiles
"""
import os
import sys

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TW, "pipeline"))
import tier0  # noqa: E402

# every tile the tier 0 windows of sha and the 84 mainland metro rows touch (2026-10-24)
CHINA_METROS = "R4_C26 R4_C29 R5_C27 R5_C28 R5_C29 R6_C28 R6_C29 R6_C30 R7_C29 R7_C30".split()
OLD = os.path.join(TW, "data", "raw", "t1150_ghsl")
for t in sys.argv[1:] or CHINA_METROS:
    r, c = (int(x[1:]) for x in t.split("_"))
    name = f"{tier0.PRODUCTS['tot_v'][1].format(epoch=2020)}_V1_0_R{r}_C{c}.zip"
    dest = tier0.raw("ghsl", name)
    for suffix in ("", ".none"):
        if os.path.exists(os.path.join(OLD, name + suffix)) and not os.path.exists(dest + suffix):
            os.replace(os.path.join(OLD, name + suffix), dest + suffix)
    print(t, tier0._tile_path("tot_v", 2020, r, c) or "none (open ocean)", flush=True)
