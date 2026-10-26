"""T-1270: each zone's terrain classes, the ground factor on track prices (docs/track-and-service.md 6.4).

Source: GEDTM30 at 240 m (OpenGeoHub, v1.2, CC BY 4.0, Zenodo record 18887460), a bare-earth terrain
model, so city towers do not read as hills. One 10 GB cloud-optimised GeoTIFF in EPSG:4326, 1/450
degree pixels, float32 metres, read here tile by tile. Download it to `data/raw/gedtm30/`:

    curl -L -o data/raw/gedtm30/gedtm_rf_m_240m_v1.2.tif https://zenodo.org/api/records/18887460/files/gedtm_rf_m_240m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif/content

Per zone box (registry.json, widened by PAD_DEG): slope in degrees at each source pixel (central
differences in metres at the pixel's latitude), the mean slope over each cell of a world lattice of
PER_DEG cells a degree (about 460 m north-south), a class by THRESH_DEG (0 flat, 1 rolling, 2 hilly,
3 mountain), then a 3 x 3 majority filter so the classes come in patches and a route's price does not
flicker. Every zone's grid sits on the one lattice, so where boxes overlap the grids agree.

Out: `data/terrain/<id>.terrain`, the format `sim/src/track/cost.rs` `ClassGrid::decode` reads: `TWT1`,
u32 w, h, i32 col0, row0 (lattice cell of the south-west corner from 180 W, 90 S), u32 PER_DEG, then
(class u8, run length LEB128) runs over the grid row-major from the south.

    zone_terrain.py build [id ...]        every registry zone, or those named; skips files already made
    zone_terrain.py build --force [id ...]
    zone_terrain.py probe lon lat [km]    mean slope and class shares round a point (calibration)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import logging
import math
import struct
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent
TW = HERE.parent
SRC = TW / "data" / "raw" / "gedtm30" / "gedtm_rf_m_240m_v1.2.tif"
SRC_URL = "https://zenodo.org/api/records/18887460/files/gedtm_rf_m_240m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif/content"
OUT = TW / "data" / "terrain"
REGISTRY = TW / "app" / "src" / "game" / "registry.json"

PER_DEG = 240
THRESH_DEG = (2.0, 5.0, 10.0)
PAD_DEG = 0.03
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0
TILE_CACHE = 24
# the file's nodata (float32 max) is written as a double tifffile cannot cast; the code masks it
logging.getLogger("tifffile").setLevel(logging.ERROR)


class Dem:
    """The source raster's full-resolution level, read by tiles with a small cache."""

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


def lattice_box(w: float, s: float, e: float, n: float) -> tuple[int, int, int, int]:
    """Lattice columns and rows (from 180 W, 90 S) covering a box: col0, row0, width, height."""
    c0 = math.floor((w + 180.0) * PER_DEG)
    c1 = math.ceil((e + 180.0) * PER_DEG)
    r0 = math.floor((s + 90.0) * PER_DEG)
    r1 = math.ceil((n + 90.0) * PER_DEG)
    return c0, r0, c1 - c0, r1 - r0


def mean_slopes(dem: Dem, c0: int, r0: int, cw: int, ch: int) -> np.ndarray:
    """Mean slope in degrees over each lattice cell of the box, rows from the south; NaN where the
    source has nothing (the sea, beyond 85 N or 65 S)."""
    w = c0 / PER_DEG - 180.0
    s = r0 / PER_DEG - 90.0
    e = (c0 + cw) / PER_DEG - 180.0
    n = (r0 + ch) / PER_DEG - 90.0
    px = dem.px
    # source pixels whose centres fall in the box, plus one each way for the differences
    pc0 = math.floor((w - dem.lon0) / px) - 1
    pc1 = math.ceil((e - dem.lon0) / px) + 1
    pr0 = math.floor((dem.lat0 - n) / px) - 1
    pr1 = math.ceil((dem.lat0 - s) / px) + 1
    z = dem.window(pr0, pr1, pc0, pc1)
    lats = dem.lat0 - (np.arange(pr0, pr1) + 0.5) * px
    lons = dem.lon0 + (np.arange(pc0, pc1) + 0.5) * px
    dy = px * M_PER_DEG_LAT
    dx = px * M_PER_DEG_LON * np.cos(np.radians(lats))[:, None]
    gy, gx = np.gradient(z)
    slope = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy)))
    slope = slope[1:-1, 1:-1]
    lats, lons = lats[1:-1], lons[1:-1]
    rows = np.floor((lats + 90.0) * PER_DEG).astype(np.int64) - r0
    cols = np.floor((lons + 180.0) * PER_DEG).astype(np.int64) - c0
    ok = np.isfinite(slope) & ((rows >= 0) & (rows < ch))[:, None] & ((cols >= 0) & (cols < cw))[None, :]
    idx = (rows[:, None] * cw + cols[None, :])[ok]
    total = np.bincount(idx, weights=slope[ok], minlength=cw * ch)
    count = np.bincount(idx, minlength=cw * ch)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / count
    return mean.reshape(ch, cw)


def classes_of(mean: np.ndarray) -> np.ndarray:
    """Classes by THRESH_DEG, then a 3 x 3 majority filter (ties keep the cell's own class)."""
    m = np.nan_to_num(mean, nan=0.0)
    cls = np.digitize(m, THRESH_DEG).astype(np.uint8)
    counts = np.stack([uniform_filter((cls == k).astype(np.float32), size=3, mode="nearest") * 9 for k in range(4)])
    counts += 0.5 * np.stack([cls == k for k in range(4)])
    return counts.argmax(axis=0).astype(np.uint8)


def encode(cls: np.ndarray, c0: int, r0: int) -> bytes:
    ch, cw = cls.shape
    out = bytearray(b"TWT1")
    out += struct.pack("<IIiiI", cw, ch, c0, r0, PER_DEG)
    flat = cls.reshape(-1)
    change = np.flatnonzero(np.diff(flat)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [flat.size]))
    for a, b in zip(starts.tolist(), ends.tolist()):
        out.append(int(flat[a]))
        n = b - a
        while True:
            x = n & 0x7F
            n >>= 7
            if n:
                out.append(x | 0x80)
            else:
                out.append(x)
                break
    return bytes(out)


def zones() -> list[dict]:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [c for c in reg["cities"] if c.get("box")]


def build(ids: list[str], force: bool) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dem = Dem(SRC)
    todo = [z for z in zones() if not ids or z["id"] in ids]
    # neighbouring boxes one after another, so the tile cache serves them
    todo.sort(key=lambda z: (round(z["box"][3] / 4), z["box"][0]))
    t0 = time.time()
    made = 0
    size = 0
    shares = np.zeros(4)
    for k, z in enumerate(todo):
        path = OUT / f"{z['id']}.terrain"
        if path.exists() and not force:
            continue
        w, s, e, n = z["box"]
        c0, r0, cw, ch = lattice_box(w - PAD_DEG, s - PAD_DEG, e + PAD_DEG, n + PAD_DEG)
        cls = classes_of(mean_slopes(dem, c0, r0, cw, ch))
        data = encode(cls, c0, r0)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        made += 1
        size += len(data)
        shares += np.bincount(cls.reshape(-1), minlength=4)
        if made % 50 == 0 or k == len(todo) - 1:
            print(f"{k + 1}/{len(todo)} {z['id']} {len(data) / 1024:.1f} KB, {time.time() - t0:.0f} s", flush=True)
    if made:
        sh = shares / shares.sum()
        print(f"made {made} files, {size / 1e6:.1f} MB, {time.time() - t0:.0f} s; cells flat {sh[0]:.1%} rolling {sh[1]:.1%} hilly {sh[2]:.1%} mountain {sh[3]:.1%}")
    else:
        print("nothing to make")


def probe(lon: float, lat: float, km: float) -> None:
    dem = Dem(SRC)
    dlat = km / 110.574
    dlon = dlat / math.cos(math.radians(lat))
    c0, r0, cw, ch = lattice_box(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    mean = mean_slopes(dem, c0, r0, cw, ch)
    cls = classes_of(mean)
    m = mean[np.isfinite(mean)]
    q = np.percentile(m, [10, 50, 90]) if m.size else [math.nan] * 3
    sh = np.bincount(cls.reshape(-1), minlength=4) / cls.size
    print(f"{lon:.3f},{lat:.3f} r {km:g} km: slope p10 {q[0]:.1f} p50 {q[1]:.1f} p90 {q[2]:.1f}; flat {sh[0]:.0%} rolling {sh[1]:.0%} hilly {sh[2]:.0%} mountain {sh[3]:.0%}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "build":
        force = "--force" in args
        build([a for a in args[1:] if a != "--force"], force)
    elif args and args[0] == "probe":
        probe(float(args[1]), float(args[2]), float(args[3]) if len(args) > 3 else 5.0)
    else:
        print(__doc__)
