"""T-100 / T-101 point checks: MERIT Hydro (via Hydrography90m) length and drop between two points
on a river, against RiverATLAS (T-073 reachtrips.path), published lengths and T-031's known drops.

    python points.py
"""
import os
import sys
import io
import contextlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import rasterio
from rasterio.windows import Window

sys.path.insert(0, r"C:\Users\anita\projects\maps\isochrone\notes\T-073")
with contextlib.redirect_stdout(io.StringIO()):
    import reachtrips as rt  # noqa: E402

U = "/vsicurl/https://public.igb-berlin.de/public.php/webdav/global/"
ENV = dict(GDAL_HTTP_USERPWD="agciopgzXjWswF4:", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
           CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="3")
LAYERS = {"diff": "outlet_diff_dw_basin_ovr.tif", "dist": "outlet_dist_dw_basin_ovr.tif",
          "acc": "accumulation_ovr.tif", "basin": "basin_ovr.tif"}
R = 8

# (name, upstream lat/lon, downstream lat/lon or None for the basin outlet, published km, real drop m)
CHECKS = [
    ("Mississippi Itasca - Knutson Dam", (47.2405, -95.2075), (47.449, -94.482), 134.2, None),
    ("Beaverhead+Jefferson", (45.006, -112.852), (45.925, -111.515), 244.6, None),
    ("Jefferson from Big Hole mouth", (45.560, -112.335), (45.925, -111.515), 133.6, None),
    ("Illinois RM 273-0", (41.385, -88.262), (38.968, -90.440), 439.4, None),
    ("Ohio RM 0-981", (40.448, -80.030), (36.990, -89.130), 1578.7, None),
    ("Missouri Three Forks - mouth", (45.935, -111.500), (38.815, -90.120), 3767.4, None),
    ("Amazon Manaus - sea", (-3.14, -59.90), None, None, 20),
    ("Congo Kisangani - Kinshasa", (0.50, 25.15), (-4.30, 15.32), None, 110),
    ("Congo Kinshasa - sea", (-4.30, 15.32), None, None, 275),
    ("Mississippi St Louis - sea", (38.63, -90.18), None, None, 116),
    ("Danube Vienna - sea", (48.22, 16.40), None, None, 155),
    ("Colorado Lees Ferry - Pearce Ferry", (36.865, -111.588), (36.115, -113.99), None, 580),
    ("Indus Skardu - Tarbela", (35.30, 75.63), (34.09, 72.70), None, 1750),
    ("Glomma Elverum - sea", (60.88, 11.56), None, None, 187),
]


class H90:
    def __init__(self):
        self.env = rasterio.Env(**ENV)
        self.env.__enter__()
        self.ds = {k: rasterio.open(U + v) for k, v in LAYERS.items()}

    def at(self, lat, lon, rad=R):
        c = int(np.floor((lon + 180) * 1200)) - rad
        r = int(np.floor((85 - lat) * 1200)) - rad
        win = Window(c, r, 2 * rad + 1, 2 * rad + 1)
        a = {k: d.read(1, window=win) for k, d in self.ds.items()}
        acc = np.where(a["acc"] > -1, a["acc"], -1)
        j = np.argmax(acc)
        return {k: float(v.ravel()[j]) for k, v in a.items()}


def snap_big(lat, lon, qmin, box=0.15):
    m = (np.abs(rt.co[:, 1] - lat) < box) & (np.abs(rt.co[:, 0] - lon) < box)
    idx = np.nonzero(m)[0]
    idx = idx[rt.ok[rt.vreach[idx]] & (rt.r["q"][rt.vreach[idx]] >= qmin)]
    d = (rt.co[idx, 1] - lat) ** 2 + ((rt.co[idx, 0] - lon) * np.cos(np.radians(lat))) ** 2
    j = idx[np.argmin(d)]
    return rt.vreach[j], j


def main():
    h = H90()
    rt.snap = lambda lat, lon, box=0.15: snap_big(lat, lon, QMIN, box)
    for name, a, b, pub, drop in CHECKS:
        global QMIN
        QMIN = 500 if drop else 1
        try:
            run(h, name, a, b, pub, drop)
        except (SystemExit, ValueError) as e:
            print(name, "failed:", e, flush=True)


QMIN = 1


def run(h, name, a, b, pub, drop):
    if True:
        ka, ja = rt.snap(*a)
        pa = rt.co[ja]
        ma = h.at(pa[1], pa[0])
        if b is None:
            mb = {"diff": 0.0, "dist": 0.0, "basin": ma["basin"], "acc": np.nan}
            ra = rt.r["dist_up"][ka] if False else None
            # RiverATLAS length to the mouth: follow down to the end
            km, k = 0.0, ka
            frac = (rt.off[ka + 1] - ja - 1) / max(rt.off[ka + 1] - rt.off[ka] - 1, 1)
            km = rt.r["length_km"][ka] * frac
            k = rt.down[ka]
            while k >= 0:
                km += rt.r["length_km"][k]
                k = rt.down[k]
            ra = km
        else:
            kb, jb = rt.snap(*b)
            pb = rt.co[jb]
            mb = h.at(pb[1], pb[0])
            ra = sum(x[1] for x in rt.path(a, b))
        mkm = (ma["dist"] - mb["dist"]) / 1000
        md = ma["diff"] - mb["diff"]
        same = ma["basin"] == mb["basin"]
        s = "%-36s RA %7.0f km  MERIT %7.0f km  M/RA %.2f" % (name, ra, mkm, mkm / ra)
        if pub:
            s += "  published %6.0f  pub/RA %.2f  pub/M %.2f" % (pub, pub / ra, pub / mkm)
        if drop:
            s += "  MERIT drop %5.0f m (real ~%d)" % (md, drop)
        s += "  acc %.0f/%.0f km2%s" % (ma["acc"], mb["acc"], "" if same else "  BASIN DIFFERS")
        print(s, flush=True)


if __name__ == "__main__":
    main()
