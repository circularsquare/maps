"""T-162: a US pack's zones against LODES OD, for t314_loo.py's US setup. Read-only on the packs;
LODES is the fit target only, never a runtime input.

Observed: LODES 8 OD 2023 S000, home and work blocks both in the pack's counties, each block in
the zone of its internal point and its state. `P`, `Q`: each zone's homes and jobs by county.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import build_city as bc  # noqa: E402
import fit_decay as fd  # noqa: E402
from read_pack import load_pack  # noqa: E402

BAND_KM, MAX_KM = 2.0, 120.0


class City:
    def __init__(self, city, packs=bc.PACKS):
        t0 = time.time()
        header, a = load_pack(city, packs)
        g = header["gravity"]
        self.name = city
        zm = g["zone_m"]
        x0, y0 = g["grid_origin_m"]
        w, h = g["grid"]
        gx, gy = a["zone_gx"].astype(np.int64), a["zone_gy"].astype(np.int64)
        nz = len(gx)
        zr = a["zone_region"].astype(np.int64) if "zone_region" in a else np.zeros(nz, np.int64)
        ids = header.get("regions", {}).get("ids", ["?"])
        self.zr = zr
        zk = zm / 1000.0
        dx = (gx[:, None] - gx[None, :]).astype(np.float32)
        dy = (gy[:, None] - gy[None, :]).astype(np.float32)
        self.d = np.float32(zk) * np.sqrt(dx * dx + dy * dy)
        del dx, dy
        self.d[self.d == 0] = np.float32(0.52 * zk)
        self.band = np.minimum(self.d / BAND_KM, MAX_KM / BAND_KM - 1).astype(np.int16)
        # blocks -> zones: square of the internal point, the block's own state where that square has it
        b = pd.read_csv(os.path.join(bc.WORK, f"{city}.blocks.csv.gz"), dtype={"geoid": str},
                        usecols=["geoid", "lat", "lon", "pop", "jobs"])
        lon0, lat0 = header["origin"]["lon"], header["origin"]["lat"]
        bx, by = fd.local_xy(b["lat"].to_numpy(), b["lon"].to_numpy(), lon0, lat0)
        bgx = np.floor((bx - x0) / zm).astype(np.int64)
        bgy = np.floor((by - y0) / zm).astype(np.int64)
        rid = {c: i for i, c in enumerate(ids)}
        breg = b["geoid"].str[:2].map(rid).fillna(-1).astype(np.int64).to_numpy()
        lookup = {}
        anyz = {}
        for i in range(nz):
            lookup[(gx[i] * h + gy[i], zr[i])] = i
            anyz.setdefault(gx[i] * h + gy[i], i)
        sq = bgx * h + bgy
        inside = (bgx >= 0) & (bgx < w) & (bgy >= 0) & (bgy < h)
        bzone = np.array([lookup.get((s, r), anyz.get(s, -1)) if ok else -1 for s, r, ok in zip(sq, breg, inside)], np.int64)
        self.blocks = b
        self.bzone = bzone
        self.bstate = b["geoid"].str[:2].to_numpy()
        counties = sorted(bc.CITIES[city]["counties"])
        self.counties = counties
        cidx = {c: i for i, c in enumerate(counties)}
        bcounty = b["geoid"].str[:5].map(cidx).to_numpy()
        self.bcounty = bcounty
        nc = len(counties)
        ok = bzone >= 0
        P = np.zeros((nz, nc))
        Q = np.zeros((nz, nc))
        np.add.at(P, (bzone[ok], bcounty[ok]), b["pop"].to_numpy()[ok])
        np.add.at(Q, (bzone[ok], bcounty[ok]), b["jobs"].to_numpy()[ok])
        self.P = (P / np.maximum(P.sum(1, keepdims=True), 1e-9)).astype(np.float32)
        self.Q = (Q / np.maximum(Q.sum(1, keepdims=True), 1e-9)).astype(np.float32)
        print(f"{city}: {nz:,} zones ({len(set(gx * h + gy)):,} squares), regions {ids}; "
              f"{int((~ok).sum()):,} blocks outside every zone; {time.time() - t0:.0f} s", flush=True)
        self.observe()

    def observe(self):
        od = fd.lodes_od(self.name)
        bi = {g: i for i, g in enumerate(self.blocks["geoid"])}
        hi = od["h_geocode"].map(bi)
        wi = od["w_geocode"].map(bi)
        miss = hi.isna() | wi.isna()
        od, hi, wi = od[~miss], hi[~miss].to_numpy(np.int64), wi[~miss].to_numpy(np.int64)
        n = od["S000"].to_numpy(np.float64)
        zh, zw = self.bzone[hi], self.bzone[wi]
        ok = (zh >= 0) & (zw >= 0)
        self.obs_cross = float(n[self.bstate[hi] != self.bstate[wi]].sum() / n.sum())
        tl = np.bincount(self.band[zh[ok], zw[ok]], weights=n[ok], minlength=int(MAX_KM / BAND_KM))
        self.obs_tl = tl / tl.sum()
        nc = len(self.counties)
        C = np.zeros((nc, nc))
        np.add.at(C, (self.bcounty[hi], self.bcounty[wi]), n)
        self.obs_C = C / C.sum()
        self.obs_mean = float((self.d[zh[ok], zw[ok]] * n[ok]).sum() / n[ok].sum())
        print(f"  LODES: {n.sum():,.0f} jobs, between states {self.obs_cross:.2%}, own county {np.trace(self.obs_C):.1%}, "
              f"mean zone distance {self.obs_mean:.2f} km", flush=True)
