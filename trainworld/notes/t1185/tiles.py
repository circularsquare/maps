"""T-1185 rebuild, step 1: the 14 GHS-BUILT-V total (E2020) tiles China's zones need beyond the
metros' ten, moved from data/raw/t1185_ghsl/ (fetched 2026-10-25, 154 MB) into data/raw/tier0/ghsl/,
where tier0._tile_path looks; any missing one downloads through tier0 itself. Then every tile the
plan's Chinese zone rows touch is checked present.

    python trainworld/notes/t1185/tiles.py
"""
import json
import os
import sys

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TW, "pipeline"))
import tier0  # noqa: E402

NEW = "R3_C27 R3_C28 R3_C29 R4_C25 R4_C27 R4_C28 R4_C30 R5_C25 R5_C26 R5_C30 R6_C25 R6_C26 R6_C27 R7_C28".split()
OLD = os.path.join(TW, "data", "raw", "t1185_ghsl")
product = tier0.PRODUCTS["tot_v"][1].format(epoch=2020)


def tiles(bbox):
    """The 1,000 km tiles a lon/lat box's window touches (tier0.ghsl_window's arithmetic)."""
    fwd, _ = tier0._moll()
    w, s, e, n = bbox
    xs, ys = fwd.transform([w, w, e, e, (w + e) / 2, (w + e) / 2], [s, n, s, n, s, n])
    r0, r1 = (int((tier0.MOLL_Y0 - y) // tier0.TILE_M) + 1 for y in (max(ys), min(ys)))
    c0, c1 = (int((x - tier0.MOLL_X0) // tier0.TILE_M) + 1 for x in (min(xs), max(xs)))
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


for t in NEW:
    r, c = (int(x[1:]) for x in t.split("_"))
    name = f"{product}_V1_0_R{r}_C{c}.zip"
    dest = tier0.raw("ghsl", name)
    for suffix in ("", ".none"):
        if os.path.exists(os.path.join(OLD, name + suffix)) and not os.path.exists(dest + suffix):
            os.replace(os.path.join(OLD, name + suffix), dest + suffix)
    print(t, tier0._tile_path("tot_v", 2020, r, c) or "none (open ocean)", flush=True)
with open(os.path.join(TW, "pipeline", "world_plan.json"), encoding="utf-8") as f:
    rows = [r for r in json.load(f)["metros"] if r.get("kind") == "zone" and "CHN" in (r.get("units") or {})]
need = set().union(*(tiles(r["bbox"]) for r in rows))
miss = [f"R{r}_C{c}" for r, c in sorted(need) if not os.path.exists(tier0.raw("ghsl", f"{product}_V1_0_R{r}_C{c}.zip"))
        and not os.path.exists(tier0.raw("ghsl", f"{product}_V1_0_R{r}_C{c}.zip.none"))]
print(f"{len(rows)} zone rows touch {len(need)} tiles; missing: {' '.join(miss) or 'none'}")
if not miss and os.path.isdir(OLD) and not os.listdir(OLD):
    os.rmdir(OLD)
