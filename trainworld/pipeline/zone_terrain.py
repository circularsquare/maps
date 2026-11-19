"""T-1270, T-1854: the terrain classes, the ground factor on track prices (docs/track-and-service.md 6.4), and
the `.terrain` format. `world_terrain.py` makes the game's files with them, one per degree block (T-2901).

Source: the fine terrain intermediate (`terrain_fine.py`, `data/work/terrain-fine-world/`): the mean slope of
GEDTM30's 30 m bare-earth pixels over each cell of the world lattice of PER_DEG cells a degree (about
115 m north-south). A class by THRESH_Q (0 flat, 1 rolling, 2 hilly, 3 mountain; no data counts flat), then a
3 x 3 majority filter (ties keep the cell's own class), which drops single-cell specks of the 30 m noise so a
route's price does not flicker cell to cell but keeps a valley floor two cells (about 230 m) wide, then capped
by the coarse ground round it (COARSE_THRESH_DEG on the 1/240 degree slope taken bilinearly at the cell's
centre, against towers and raised roads; T-3372). THRESH_Q was set so the world's mix of classes over the
zones stayed what the 460 m classes had (notes/T-1854.md).

The format (`encode`), which `sim/src/track/cost.rs` `ClassGrid::decode` reads: gzip of `TWT1`, u32 w, h,
i32 col0, row0 (lattice cell of the south-west corner from 180 W, 90 S), u32 PER_DEG, then (class u8, run
length LEB128) runs over the grid row-major from the south. The clock worker unzips.

    zone_terrain.py mix [dir]             the class shares over every `.terrain` file in dir (the world's blocks)
    zone_terrain.py probe lon lat [km]    mean slope and class shares round a point

`Dem` reads GEDTM30 at 240 m (data/raw/gedtm30/, see T-1270.md) for market_drives.py.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import gzip
import logging
import math
import struct
import subprocess
import sys
import tarfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent
TW = HERE.parent
SRC = TW / "data" / "raw" / "gedtm30" / "gedtm_rf_m_240m_v1.2.tif"
SRC_URL = "https://zenodo.org/api/records/18887460/files/gedtm_rf_m_240m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif/content"
FINE = TW / "data" / "work" / "terrain-fine-world"
BLOCKS = TW / "data" / "terrain-world-v2"

PER_DEG = 960
# class bounds on the fine cells' mean 30 m slope, in quarter degrees (the intermediate's unit):
# rolling from 2.5 degrees, hilly from 5.5, mountain from 11 (notes/T-1854.md), set so the shares of flat,
# rolling, hilly and mountain ground stayed the 460 m classes' (T-1270) over the 1,243 zones that had them,
# 0.5888, 0.1617, 0.1081, 0.1414 by area (Anita, 2026-10-30)
THRESH_Q = (10, 22, 44)
# a fine cell's class is at most the class of the ground round it at 1/240 degree (the slope between
# neighbouring mean elevations, about 920 m apart, taken bilinearly at the fine cell's centre, by
# COARSE_THRESH_DEG): 30 m bare earth still carries
# towers and raised roads, which read as hills in dense centres (uncapped, 47% of central Tokyo was
# rolling or hilly); set as low as keeps Tokyo, Nagoya and Shanghai's centres flat (notes/T-1854.md)
COARSE_PER_DEG = 240
COARSE_THRESH_DEG = (1.25, 2.5, 5.0)
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0
TILE_CACHE = 24
# the file's nodata (float32 max) is written as a double tifffile cannot cast; the code masks it
logging.getLogger("tifffile").setLevel(logging.ERROR)


class Dem:
    """The 240 m source raster's full-resolution level, read by tiles with a small cache."""

    def __init__(self, path: Path):
        self.tif = tifffile.TiffFile(str(path))
        self.size = path.stat().st_size
        self.page = self.tif.pages[0]
        p = self.page
        self.h, self.w = p.shape[:2]
        self.tw, self.th = p.tilewidth, p.tilelength
        self.across = -(-self.w // self.tw)
        scale = p.tags["ModelPixelScaleTag"].value
        tie = p.tags["ModelTiepointTag"].value
        self.px = float(scale[0])
        self.lon0, self.lat0 = float(tie[3]), float(tie[4])  # the top-left corner
        self.cache: "OrderedDict[tuple[int, int], np.ndarray]" = OrderedDict()

    def tile(self, ty: int, tx: int) -> np.ndarray:
        key = (ty, tx)
        t = self.cache.get(key)
        if t is not None:
            self.cache.move_to_end(key)
            return t
        idx = ty * self.across + tx
        off, n = self.page.dataoffsets[idx], self.page.databytecounts[idx]
        if off + n <= self.size:
            fh = self.tif.filehandle
            fh.seek(off)
            data = fh.read(n)
        else:
            # a download still under way: this tile's bytes from Zenodo by range
            data = subprocess.run(["curl", "-L", "-s", "-S", "--retry", "5", "-r", f"{off}-{off + n - 1}", SRC_URL],
                                  capture_output=True, check=True).stdout
            if len(data) != n:
                raise RuntimeError(f"tile {idx}: {len(data)} of {n} bytes")
        seg, _, shape = self.page.decode(data, idx, jpegtables=self.page.jpegtables)
        t = np.asarray(seg, dtype=np.float32).reshape(shape[-3], shape[-2])
        t = np.where((t > 1e30) | (t < -1000), np.nan, t)
        self.cache[key] = t
        if len(self.cache) > TILE_CACHE:
            self.cache.popitem(last=False)
        return t

    def window(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """Rows r0..r1 and columns c0..c1 (exclusive ends) of the full raster; outside it NaN."""
        out = np.full((r1 - r0, c1 - c0), np.nan, dtype=np.float32)
        rr0, rr1, cc0, cc1 = max(r0, 0), min(r1, self.h), max(c0, 0), min(c1, self.w)
        if rr0 >= rr1 or cc0 >= cc1:
            return out
        for ty in range(rr0 // self.th, (rr1 - 1) // self.th + 1):
            for tx in range(cc0 // self.tw, (cc1 - 1) // self.tw + 1):
                t = self.tile(ty, tx)
                a0, a1 = max(rr0, ty * self.th), min(rr1, (ty + 1) * self.th)
                b0, b1 = max(cc0, tx * self.tw), min(cc1, (tx + 1) * self.tw)
                out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = t[a0 - ty * self.th:a1 - ty * self.th, b0 - tx * self.tw:b1 - tx * self.tw]
        return out


class Fine:
    """The intermediate's mean-slope planes, by one-degree block, read from its part tars."""

    def __init__(self, root: Path = FINE, cache: int = 300):
        from terrain_fine import decode_block
        self.decode = decode_block
        self.index = {}
        # TW_FINE_OVER: a folder of a test run's part tars whose blocks stand in for the intermediate's
        over = [Path(os.environ["TW_FINE_OVER"])] if os.environ.get("TW_FINE_OVER") else []
        for p in sorted(root.glob("part*.tar")) + [q for o in over for q in sorted(o.glob("part*.tar"))]:
            with tarfile.open(p) as tf:
                for m in tf.getmembers():
                    self.index[m.name] = (p, m.offset_data, m.size)
        if not self.index:
            raise SystemExit(f"no part*.tar in {root}: the intermediate is missing (terrain_fine.py)")
        self.cache: "OrderedDict[tuple[int, int], np.ndarray | None]" = OrderedDict()
        self.size = cache

    def block(self, la: int, lo: int):
        key = (la, lo)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        hit = self.index.get(f"{la}_{lo}.twf")
        plane = None
        if hit:
            p, off, n = hit
            with open(p, "rb") as f:
                f.seek(off)
                plane = self.decode(f.read(n))[2]
        self.cache[key] = plane
        if len(self.cache) > self.size:
            self.cache.popitem(last=False)
        return plane

    def slopes(self, c0: int, r0: int, cw: int, ch: int) -> np.ndarray:
        """Mean slope in quarter degrees (255 no data) over lattice cells c0.., r0.. (rows from the south)."""
        return self._plane(0, PER_DEG, 255, np.uint8, c0, r0, cw, ch)

    def elev(self, c0: int, r0: int, cw: int, ch: int) -> np.ndarray:
        """Mean elevation in metres (-32768 no data) over cells of the 1/240 degree lattice."""
        return self._plane(1, COARSE_PER_DEG, -32768, np.int16, c0, r0, cw, ch)

    def _plane(self, k, per, fill, dtype, c0, r0, cw, ch):
        out = np.full((ch, cw), fill, dtype=dtype)
        for la in range(math.floor(r0 / per) - 90, math.floor((r0 + ch - 1) / per) - 90 + 1):
            for lo in range(math.floor(c0 / per) - 180, math.floor((c0 + cw - 1) / per) - 180 + 1):
                b = self.block(la, lo)
                if b is None:
                    continue
                b = b[k]
                br, bc = (la + 90) * per, (lo + 180) * per
                a0, a1 = max(r0, br), min(r0 + ch, br + per)
                b0, b1 = max(c0, bc), min(c0 + cw, bc + per)
                out[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = b[a0 - br:a1 - br, b0 - bc:b1 - bc]
        return out

    def coarse_slope(self, k0: int, j0: int, kw: int, jh: int) -> np.ndarray:
        """Slope in degrees between neighbouring 1/240 degree mean elevations, at the coarse cells
        k0.., j0.. (no data 0)."""
        e = self.elev(k0 - 1, j0 - 1, kw + 2, jh + 2).astype(np.float32)
        e[e == -32768] = np.nan
        lat = (j0 - 1 + np.arange(jh + 2) + 0.5) / COARSE_PER_DEG - 90
        dy = M_PER_DEG_LAT / COARSE_PER_DEG
        dx = (M_PER_DEG_LON / COARSE_PER_DEG * np.cos(np.radians(lat)))[:, None]
        gy, gx = np.gradient(e)
        s = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy)))
        return np.nan_to_num(s, nan=0.0)[1:-1, 1:-1]

    def coarse(self, c0: int, r0: int, cw: int, ch: int, thresh=None) -> np.ndarray:
        """Classes by COARSE_THRESH_DEG of the coarse slope (`coarse_slope`) taken bilinearly at each
        fine cell's centre, on the fine cells c0.., r0.. (T-3372: the cap's borders follow the ground
        at the fine cells instead of making 460 m squares)."""
        f = PER_DEG // COARSE_PER_DEG
        k0, j0 = c0 // f - 1, r0 // f - 1
        k1, j1 = (c0 + cw - 1) // f + 2, (r0 + ch - 1) // f + 2
        s = self.coarse_slope(k0, j0, k1 - k0, j1 - j0)
        # a fine cell's centre in coarse cell units, from the centre of coarse cell k0 (j0)
        v = (r0 + np.arange(ch) + 0.5) / f - 0.5 - j0
        u = (c0 + np.arange(cw) + 0.5) / f - 0.5 - k0
        jv, ku = np.floor(v).astype(int), np.floor(u).astype(int)
        fv, fu = (v - jv)[:, None].astype(np.float32), (u - ku)[None, :].astype(np.float32)
        a, b = s[jv[:, None], ku[None, :]], s[jv[:, None], ku[None, :] + 1]
        c, d = s[jv[:, None] + 1, ku[None, :]], s[jv[:, None] + 1, ku[None, :] + 1]
        top = a + (b - a) * fu
        sl = top + (c + (d - c) * fu - top) * fv
        return np.digitize(sl, thresh or COARSE_THRESH_DEG).astype(np.uint8)


def lattice_box(w: float, s: float, e: float, n: float) -> tuple[int, int, int, int]:
    """Lattice columns and rows (from 180 W, 90 S) covering a box: col0, row0, width, height."""
    c0 = math.floor((w + 180.0) * PER_DEG)
    c1 = math.ceil((e + 180.0) * PER_DEG)
    r0 = math.floor((s + 90.0) * PER_DEG)
    r1 = math.ceil((n + 90.0) * PER_DEG)
    return c0, r0, c1 - c0, r1 - r0


def classes_of(q: np.ndarray, cap: np.ndarray, thresh=THRESH_Q) -> np.ndarray:
    """Classes by `thresh` (no data flat), then a 3 x 3 majority filter (ties keep the cell's own
    class), then at most `cap` (`Fine.coarse`)."""
    cls = np.digitize(np.where(q == 255, 0, q), thresh).astype(np.uint8)
    best = np.zeros(cls.shape, np.float32)
    out = np.zeros(cls.shape, np.uint8)
    for k in range(4):
        m = cls == k
        c = uniform_filter(m.astype(np.float32), size=3, mode="nearest") * 9 + 0.5 * m
        take = c > best
        out[take] = k
        best[take] = c[take]
    return np.minimum(out, cap)


def encode(cls: np.ndarray, c0: int, r0: int) -> bytes:
    ch, cw = cls.shape
    out = bytearray(b"TWT1")
    out += struct.pack("<IIiiI", cw, ch, c0, r0, PER_DEG)
    flat = cls.reshape(-1)
    change = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], change))
    runs = np.diff(np.concatenate((starts, [flat.size])))
    vals = flat[starts]
    for v, n in zip(vals.tolist(), runs.tolist()):
        out.append(v)
        while True:
            x = n & 0x7F
            n >>= 7
            if n:
                out.append(x | 0x80)
            else:
                out.append(x)
                break
    return gzip.compress(bytes(out), compresslevel=9, mtime=0)


def decode(b: bytes) -> tuple[np.ndarray, int, int, int]:
    """A `.terrain` file (gzip or not): classes (rows from the south), col0, row0, per_deg."""
    if b[:2] == b"\x1f\x8b":
        b = gzip.decompress(b)
    w, h, c0, r0, per = struct.unpack("<IIiiI", b[4:24])
    out = np.zeros(w * h, np.uint8)
    i, k = 24, 0
    while i < len(b) and k < w * h:
        c = b[i]
        i += 1
        run = shift = 0
        while True:
            x = b[i]
            i += 1
            run |= (x & 0x7F) << shift
            if not x & 0x80:
                break
            shift += 7
        out[k:k + run] = c
        k += run
    return out.reshape(h, w), c0, r0, per


_FINE = None


def _fine() -> Fine:
    global _FINE
    if _FINE is None:
        _FINE = Fine()
    return _FINE


def mix(path: Path) -> None:
    """Class shares over every zone file in `path`, cells weighted by area (cos latitude)."""
    tot = np.zeros(4)
    files = sorted(path.glob("*.terrain"))
    for f in files:
        cls, c0, r0, per = decode(f.read_bytes())
        lat = (r0 + np.arange(cls.shape[0]) + 0.5) / per - 90
        wgt = np.cos(np.radians(lat)) / per / per
        for k in range(4):
            tot[k] += ((cls == k).sum(axis=1) * wgt).sum()
    sh = tot / tot.sum()
    print(f"{len(files)} files: flat {sh[0]:.2%} rolling {sh[1]:.2%} hilly {sh[2]:.2%} mountain {sh[3]:.2%}")


def probe(lon: float, lat: float, km: float) -> None:
    dlat = km / 110.574
    dlon = dlat / math.cos(math.radians(lat))
    c0, r0, cw, ch = lattice_box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    q = _fine().slopes(c0, r0, cw, ch)
    cls = classes_of(q, _fine().coarse(c0, r0, cw, ch))
    m = q[q != 255] / 4
    p = np.percentile(m, [10, 50, 90]) if m.size else [math.nan] * 3
    sh = np.bincount(cls.reshape(-1), minlength=4) / cls.size
    print(f"{lon:.3f},{lat:.3f} r {km:g} km: slope p10 {p[0]:.1f} p50 {p[1]:.1f} p90 {p[2]:.1f}; flat {sh[0]:.0%} rolling {sh[1]:.0%} hilly {sh[2]:.0%} mountain {sh[3]:.0%}")


if __name__ == "__main__":
    rest = sys.argv[1:]
    if rest and rest[0] == "mix":
        mix(Path(rest[1]) if len(rest) > 1 else BLOCKS)
    elif rest and rest[0] == "probe":
        probe(float(rest[1]), float(rest[2]), float(rest[3]) if len(rest) > 3 else 5.0)
    else:
        print(__doc__)
