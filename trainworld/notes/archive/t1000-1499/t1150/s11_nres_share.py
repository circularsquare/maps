"""T-1150: GHSL's non-residential share of built volume at named centres (within 1 km)."""
import sys
sys.stdout.reconfigure(encoding="utf-8")
from t1150lib import *
import tier0

HERE = os.path.dirname(os.path.abspath(__file__))
orig = tier0._tile_path
def scratch_path(kind, epoch, r, c):
    if kind == "tot_v":
        p = os.path.join(HERE, "dl", f"GHS_BUILT_V_E{epoch}_GLOBE_R2023A_54009_100_V1_0_R{r}_C{c}.zip")
        return p if os.path.exists(p) else None
    return orig(kind, epoch, r, c)
tier0._tile_path = scratch_path
spots = dict(a.split("=") for a in sys.argv[1:]) if len(sys.argv) > 1 else {}
spots = {k: tuple(map(float, v.split(","))) for k, v in spots.items()}
fwd, inv = tier0._moll()
print("spot | total volume Mm3 | non-residential share | GHS-POP people | people per 1,000 residential m3")
for name, (lo, la) in spots.items():
    box = (lo - 0.012, la - 0.01, lo + 0.012, la + 0.01)
    g = {k: tier0.ghsl_window(k, box)[0] for k in ("pop", "nres_v", "tot_v")}
    _, X0, Y0 = tier0.ghsl_window("pop", box)
    H, W = g["pop"].shape
    xs = X0 + (np.arange(W) + 0.5) * 100
    ys = Y0 - (np.arange(H) + 0.5) * 100
    xx, yy = np.meshgrid(xs, ys)
    cx, cy = fwd.transform(lo, la)
    s = np.hypot(xx - cx, yy - cy) < 1000
    tv, nv, p = g["tot_v"][s].sum(), g["nres_v"][s].sum(), g["pop"][s].sum()
    print(f"{name} | {tv / 1e6:,.1f} | {nv / max(tv, 1):.0%} | {p:,.0f} | {p / max(tv - nv, 1) * 1000:.2f}")
