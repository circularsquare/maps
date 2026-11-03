"""T-1854: the fine terrain intermediate, slope summarised from GEDTM30 at 30 m into 1/960 degree cells.

The 30 m raster is 432 GB and lives on OpenGeoHub's server; this reads only the 2048 px tiles under
the zone boxes (registry, the plan's unbuilt metros, and the next countries in NEXT_BOXES), by HTTP
range, and keeps the mean slope of each 1/960 degree cell (about 115 m) and the mean elevation of each
1/240 degree cell. Later work (thresholds, smoothing, cell sizes 115-460 m, hexes) starts from these
files, never from the raster. Only those two planes: the 30 m slope is noisy enough that a plane costs
about 6 bits a cell even compressed, and the home disk is small (notes/T-1854.md).

Source: GEDTM30 v1.2 (OpenGeoHub, CC BY 4.0), `SRC_URL`: float32 metres, 1/3600 degree pixels whose
centres sit on whole multiples of 1/3600 degree from 180 W and 85 N (the file's corner is half a pixel
plus 4 pixels out), deflate with horizontal differencing (predictor 2) on the 32-bit words.

Cells: the world lattice of 960 a degree, rows from 90 S, columns from 180 W. A pixel belongs to the
cell its centre falls in, so a cell holds 3 or 4 pixels each way (the pattern 4, 4, 4, 3 repeats every
15 pixels). Slope at a pixel is from central differences in metres at its latitude, as in
zone_terrain.py.

Towers (T-1881): GEDTM30 still carries tower blocks in some centres (Midtown Manhattan stands 50-100 m
above its streets, Chicago's Loop the same). Before the planes are taken, `tower_ground` cuts tower
fields down to a low envelope of the ground round them. A tower field is built-up land (GHSL built
volume at 1 km, `BUILT_URL`, CC BY 4.0) whose 30 m surface is rough, on ground whose envelope is gentle;
the rule and its numbers are at TOWER_BUILT_M. Each block is read with MARGIN pixels round it so the
filter agrees across block edges.

Block file `<lat>_<lon>.twf` (one per 1 x 1 degree block, south-west corner; none for a block with no
land under a fetched tile): `TWF1`, i16 lat, i16 lon, u16 planes, then per plane u8 kind, u16 cells a
side, u32 length and the zstd bytes of a square array, rows from the south:
  0 slope_mean u8 (kind 0), 960 a side: mean slope in quarter degrees (254 = 63.5 or more),
    255 = no data (sea, or a tile not fetched)
  1 elev_mean i16 (kind 1), 240 a side: metres, -32768 = no data; stored as differences along each row
Blocks travel in `part<k>.tar` files (stored, not unpacked, at home); `manifest.json` lists every block
done with its file name, or null when it had no land.

    terrain_fine.py blocks <blocks.json> [lat_lon ...]       the block list and the tiles each needs (or only those blocks)
    terrain_fine.py run <blocks.json> <outdir> [--jobs N] [--fetch M] [--part-gb G] [--send CMD]
    terrain_fine.py check <lat> <lon> <file.twf>             a block against rasterio's read (home)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import math
import queue
import struct
import subprocess
import sys
import tarfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path

import numpy as np
import imagecodecs
from scipy.ndimage import gaussian_filter, grey_opening, maximum_filter, uniform_filter

HERE = Path(__file__).resolve().parent
TW = HERE.parent
REGISTRY = TW / "app" / "src" / "game" / "registry.json"
PLAN = HERE / "world_plan.json"
SRC_URL = "https://s3.opengeohub.org/global/dtm/v1.2/gedtm_rf_m_30m_s_20060101_20151231_go_epsg.4326.3855_v1.2.tif"
SRC_SIZE = 432407644715
BUILT_URL = ("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_BUILT_V_GLOBE_R2023A/"
             "GHS_BUILT_V_E2020_GLOBE_R2023A_4326_30ss/V1-0/GHS_BUILT_V_E2020_GLOBE_R2023A_4326_30ss_V1_0.zip")
BUILT_TIF = "GHS_BUILT_V_E2020_GLOBE_R2023A_4326_30ss_V1_0.tif"
# where `run` keeps the GHSL file (fetched once when missing); TW_GHSL_30SS: a copy elsewhere
BUILT_PATH = Path(os.environ["TW_GHSL_30SS"]) if os.environ.get("TW_GHSL_30SS") else TW / "data" / "raw" / "ghsl" / BUILT_TIF

PER_DEG = 960
PX_PER_DEG = 3600
TILE = 2048
SRC_W, SRC_H = 1296009, 540009
# pixel (row j from the top, column i) has its centre at lon -180 + (i - 4) / 3600, lat 85 + (4 - j) / 3600
I_OFF, J_OFF = 4, 4
PAD_DEG = 0.03
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0
NODATA_U8 = 255
NODATA_I16 = -32768
# countries next in line (CLAUDE.md, 2026-10-24) and Australia, built 2026-10-30: their zones are not in
# the registry yet, so their populated extents are fetched now and need no second download
NEXT_BOXES = {"AU": [112.9, -43.8, 153.7, -10.6], "KR": [124.5, 33.0, 131.0, 38.7], "TR": [25.6, 35.8, 44.9, 42.2],
              "VN": [102.1, 8.4, 109.6, 23.4], "MX": [-117.2, 14.5, -86.7, 32.8], "CA": [-128.0, 42.0, -52.5, 54.5]}

# T-1881, tower fields (notes/T-1881.md). A pixel's cut is the product of three ramps, then spread and
# softened: built-up (GHSL gross building height at 1 km, built volume over the cell's area, at least
# TOWER_BUILT_M); rough (the rms of the 30 m surface about its 60 m smoothing, taken over about 150 m,
# from nothing at the first of TOWER_RHO_M to all at the second: towers read 8-14 m, built hillsides 3-6);
# gentle (the envelope's own slope, all below the first of TOWER_GENTLE_DEG, none above the second, so
# towers on a mountainside are left: the ground there is steep anyway). The envelope is an opening of
# half-width TOWER_ENV_M, smoothed, never above the surface; a cut pixel takes envelope + (1 - cut) x
# what stands above it.
TOWER_BUILT_M = 5.0
TOWER_RHO_M = (5.5, 8.5)
TOWER_GENTLE_DEG = (3.0, 6.0)
TOWER_ENV_M = 600.0
TOWER_SPREAD_M = 300.0
# source pixels read beyond each side of a block (about 1.9 km): most of the filter's reach, so a cut
# barely changes at a block's edge
MARGIN = 64


def block_name(la: int, lo: int) -> str:
    return f"{la}_{lo}.twf"


def block_window(la: int, lo: int) -> tuple[int, int]:
    """The top-left source pixel (j0, i0) of the 3600 x 3600 pixels whose centres fall in the block."""
    i0 = I_OFF + PX_PER_DEG * (lo + 180)
    j0 = J_OFF + 1 + PX_PER_DEG * (85 - la - 1) + 0  # centre lat 85 + (4 - j) / 3600 < la + 1, >= la
    return j0, i0


def boxes() -> list[list[float]]:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    out = [c["box"] for c in reg["cities"] if c.get("box")]
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    out += [r["bbox"] for r in plan["metros"] if r.get("bbox") and r.get("status") != "shipped"]
    return out + list(NEXT_BOXES.values())


def make_blocks(path: str, only: list[str] | None = None) -> None:
    need = set()
    blocks = set()
    m = MARGIN + 2
    for w, s, e, n in boxes():
        w, s, e, n = w - PAD_DEG, s - PAD_DEG, e + PAD_DEG, n + PAD_DEG
        c0 = max(0, math.floor((w + 180) * PX_PER_DEG) + I_OFF - m) // TILE
        c1 = min(SRC_W - 1, math.ceil((e + 180) * PX_PER_DEG) + I_OFF + m) // TILE
        r0 = max(0, math.floor((85 - n) * PX_PER_DEG) + J_OFF - m) // TILE
        r1 = min(SRC_H - 1, math.ceil((85 - s) * PX_PER_DEG) + J_OFF + m) // TILE
        for ty in range(r0, r1 + 1):
            for tx in range(c0, c1 + 1):
                need.add((ty, tx))
        for la in range(math.floor(s), math.ceil(n)):
            for lo in range(math.floor(w), math.ceil(e)):
                if -65 <= la < 85:
                    blocks.add((la, lo))
    if only:
        blocks = {b for b in blocks if f"{b[0]}_{b[1]}" in only}
    rows = []
    a = MARGIN + 1
    for la, lo in sorted(blocks, key=lambda b: (-b[0], b[1])):
        j0, i0 = block_window(la, lo)
        ts = [(ty, tx) for ty in range(max(0, j0 - a) // TILE, min(SRC_H - 1, j0 + PX_PER_DEG + a - 1) // TILE + 1)
              for tx in range(max(0, i0 - a) // TILE, min(SRC_W - 1, i0 + PX_PER_DEG + a - 1) // TILE + 1) if (ty, tx) in need]
        if ts:
            rows.append({"la": la, "lo": lo, "tiles": ts})
    used = {tuple(t) for r in rows for t in r["tiles"]}
    Path(path).write_text(json.dumps({"source": SRC_URL, "blocks": rows}), encoding="utf-8")
    print(f"{len(rows)} blocks, {len(used)} tiles")


# ---- reading the source

def tile_index(url: str = SRC_URL) -> tuple[np.ndarray, np.ndarray]:
    """Full-resolution tile offsets and byte counts, from the TIFF's first IFD (BigTIFF)."""
    import requests
    s = requests.Session()

    def rng(a, n):
        for k in range(8):
            try:
                r = s.get(url, headers={"Range": f"bytes={a}-{a + n - 1}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == n:
                    return r.content
            except Exception:
                pass
            time.sleep(2 ** k)
        raise RuntimeError(f"range {a}+{n} failed")

    head = rng(0, 16)
    assert head[:4] == b"II+\x00", head[:4]
    ifd = struct.unpack("<Q", head[8:16])[0]
    nt = struct.unpack("<Q", rng(ifd, 8))[0]
    ents = rng(ifd + 8, nt * 20)
    tags = {}
    for k in range(nt):
        tag, typ, cnt = struct.unpack("<HHQ", ents[k * 20:k * 20 + 12])
        tags[tag] = (typ, cnt, ents[k * 20 + 12:k * 20 + 20])
    out = []
    for tag in (324, 325):  # TileOffsets, TileByteCounts
        typ, cnt, val = tags[tag]
        dt = {4: "<u4", 16: "<u8"}[typ]  # LONG or LONG8
        size = np.dtype(dt).itemsize
        out.append(np.frombuffer(rng(struct.unpack("<Q", val)[0], cnt * size), dtype=dt).astype(np.int64))
    return out[0], out[1]


def decode_tile(raw: bytes) -> np.ndarray:
    words = np.frombuffer(imagecodecs.zlib_decode(raw), dtype="<u4").reshape(TILE, TILE)
    z = np.cumsum(words, axis=1, dtype=np.uint32).view(np.float32)
    return np.where((z > 1e30) | (z < -1000), np.nan, z).astype(np.float32)


def window(tiles: dict, j0: int, i0: int, h: int, w: int) -> np.ndarray:
    """Source rows j0.. and columns i0.. (h x w) from decoded tiles; NaN where a tile is absent."""
    out = np.full((h, w), np.nan, dtype=np.float32)
    for (ty, tx), t in tiles.items():
        a0, a1 = max(j0, ty * TILE), min(j0 + h, (ty + 1) * TILE)
        b0, b1 = max(i0, tx * TILE), min(i0 + w, (tx + 1) * TILE)
        if a0 < a1 and b0 < b1:
            out[a0 - j0:a1 - j0, b0 - i0:b1 - i0] = t[a0 - ty * TILE:a1 - ty * TILE, b0 - tx * TILE:b1 - tx * TILE]
    return out


class Built:
    """GHSL's built volume at 30 arc seconds (BUILT_PATH, uint32 m3 a cell, 256 px LZW tiles), read by tile."""

    def __init__(self, path: Path = BUILT_PATH):
        import tifffile
        self.tif = tifffile.TiffFile(str(path))
        p = self.tif.pages[0]
        self.page = p
        self.h, self.w = p.shape[:2]
        self.th, self.tw = p.tilelength, p.tilewidth
        self.across = -(-self.w // self.tw)
        self.px = float(p.tags["ModelPixelScaleTag"].value[0])
        tie = p.tags["ModelTiepointTag"].value
        self.lon0, self.lat0 = float(tie[3]), float(tie[4])
        self.cache: dict = {}

    def tile(self, ty: int, tx: int) -> np.ndarray:
        t = self.cache.get((ty, tx))
        if t is None:
            i = ty * self.across + tx
            fh = self.tif.filehandle
            fh.seek(self.page.dataoffsets[i])
            seg = self.page.decode(fh.read(self.page.databytecounts[i]), i)[0]
            t = np.asarray(seg).reshape(self.th, self.tw).astype(np.float32)
            t[t > 4e9] = 0  # uint32 max would be no data (the sea is 0 in this file)
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[(ty, tx)] = t
        return t

    def gross(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Gross building height in metres (volume over the cell's area) at each (lat row, lon column)."""
        r = np.clip(np.floor((self.lat0 - lat) / self.px).astype(np.int64), 0, self.h - 1)
        c = np.clip(np.floor((lon - self.lon0) / self.px).astype(np.int64), 0, self.w - 1)
        r0, r1, c0, c1 = int(r.min()), int(r.max()) + 1, int(c.min()), int(c.max()) + 1
        sub = np.zeros((r1 - r0, c1 - c0), np.float32)
        for ty in range(r0 // self.th, (r1 - 1) // self.th + 1):
            for tx in range(c0 // self.tw, (c1 - 1) // self.tw + 1):
                t = self.tile(ty, tx)
                a0, a1 = max(r0, ty * self.th), min(r1, (ty + 1) * self.th)
                b0, b1 = max(c0, tx * self.tw), min(c1, (tx + 1) * self.tw)
                sub[a0 - r0:a1 - r0, b0 - c0:b1 - c0] = t[a0 - ty * self.th:a1 - ty * self.th, b0 - tx * self.tw:b1 - tx * self.tw]
        area = (self.px * M_PER_DEG_LAT) * (self.px * M_PER_DEG_LON * np.cos(np.radians(lat)))
        return sub[(r - r0)[:, None], (c - c0)[None, :]] / area[:, None].astype(np.float32)


_BUILT = None
_BUILT_PID = None


def built_reader() -> Built | None:
    # a forked worker inherits the parent's open file and would share its read offset: each process opens its own
    global _BUILT, _BUILT_PID
    if _BUILT_PID != os.getpid():
        _BUILT, _BUILT_PID = None, os.getpid()
    if _BUILT is None and BUILT_PATH.exists():
        _BUILT = Built()
    return _BUILT


def fetch_built() -> None:
    """GHSL's 1 km built volume into BUILT_PATH, once (a 200 MB zip; only its GeoTIFF is kept)."""
    import zipfile
    import requests
    if BUILT_PATH.exists():
        return
    BUILT_PATH.parent.mkdir(parents=True, exist_ok=True)
    zp = BUILT_PATH.with_suffix(".zip.part")
    with requests.get(BUILT_URL, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(zp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    with zipfile.ZipFile(zp) as z:
        tmp = BUILT_PATH.with_suffix(".tif.part")
        with z.open(BUILT_TIF) as src, open(tmp, "wb") as dst:
            while chunk := src.read(1 << 20):
                dst.write(chunk)
    os.replace(tmp, BUILT_PATH)
    zp.unlink()


def _odd(n: float) -> int:
    return 2 * max(1, round(n)) + 1


def tower_ground(z: np.ndarray, lat: float, built: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z (rows from the north, NaN no data) with tower fields cut down (TOWER_BUILT_M), and the cut
    (0-1). lat: the window's middle latitude; built: gross building height (m) on z's grid."""
    if not (built >= TOWER_BUILT_M).any():
        return z, np.zeros(z.shape, np.float32)
    dy = M_PER_DEG_LAT / PX_PER_DEG
    dx = M_PER_DEG_LON / PX_PER_DEG * math.cos(math.radians(lat))
    sig = lambda m: (m / dy, m / dx)  # noqa: E731
    f = np.where(np.isfinite(z), z, 0).astype(np.float32)
    r = f - gaussian_filter(f, sig(60))
    rho = np.sqrt(gaussian_filter(r * r, sig(150)))
    del r
    lo, hi = TOWER_RHO_M
    t = np.clip((rho - lo) / (hi - lo), 0, 1) * (built >= TOWER_BUILT_M)
    del rho
    t = gaussian_filter(maximum_filter(t, size=(_odd(TOWER_SPREAD_M / dy), _odd(TOWER_SPREAD_M / dx))), sig(150))
    k = (_odd(TOWER_ENV_M / dy), _odd(TOWER_ENV_M / dx))
    env = uniform_filter(grey_opening(f, size=k), size=k, mode="nearest")
    gy, gx = np.gradient(uniform_filter(env, size=k, mode="nearest"), dy, dx)
    steep = np.degrees(np.arctan(np.hypot(gx, gy)))
    del gx, gy
    g0, g1 = TOWER_GENTLE_DEG
    t *= np.clip((g1 - steep) / (g1 - g0), 0, 1)
    env = np.minimum(env, f)
    out = f - t * (f - env)
    return np.where(np.isfinite(z), out, np.nan).astype(np.float32), t.astype(np.float32)


def window_latlon(j0: int, i0: int, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """Centre latitude of each row and longitude of each column of source rows j0.., columns i0.."""
    return (85 + (J_OFF - (j0 + np.arange(h))) / PX_PER_DEG, -180 + (i0 + np.arange(w) - I_OFF) / PX_PER_DEG)


STARTS = np.ceil(np.arange(PER_DEG) * 15 / 4).astype(np.int64)  # first pixel of each cell along an axis


def summarise(z: np.ndarray, la: int, built: np.ndarray | None = None) -> list[np.ndarray]:
    """z: the block's window, MARGIN + 1 pixels round its 3600 x 3600, rows from the north; built: GHSL
    gross height on the same grid (None: no tower filter). The two planes, rows from the south."""
    if built is not None:
        z = tower_ground(z, la + 0.5, built)[0]
    m = MARGIN
    z = z[m:z.shape[0] - m, m:z.shape[1] - m]
    n = PX_PER_DEG
    j = np.arange(-1, n + 1)
    lat = la + (n - 1 - j) / PX_PER_DEG
    dy = M_PER_DEG_LAT / PX_PER_DEG
    dx = (M_PER_DEG_LON / PX_PER_DEG) * np.cos(np.radians(lat)).astype(np.float32)[:, None]
    gy = (z[2:, 1:-1] - z[:-2, 1:-1]) / np.float32(2 * dy)
    gx = (z[1:-1, 2:] - z[1:-1, :-2]) / (2 * dx[1:-1])
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)[::-1]  # rows from the south
    zc = z[1:-1, 1:-1][::-1]
    ok = np.isfinite(slope)
    okz = np.isfinite(zc)

    def cellsum(a):
        return np.add.reduceat(np.add.reduceat(a, STARTS, axis=1), STARTS, axis=0)

    cnt = cellsum(ok.astype(np.float32))
    ssum = cellsum(np.where(ok, slope, 0).astype(np.float32))
    have = cnt > 0
    smean = np.where(have, ssum / np.maximum(cnt, 1), 0)
    p0 = np.where(have, np.clip(np.rint(smean * 4), 0, 254), NODATA_U8).astype(np.uint8)
    # 1/240 degree cells are 15 x 15 pixels exactly
    zr = np.where(okz, zc, 0).astype(np.float64).reshape(240, 15, 240, 15).sum(axis=(1, 3))
    zn = okz.reshape(240, 15, 240, 15).sum(axis=(1, 3))
    p1 = np.where(zn > 0, np.clip(np.rint(zr / np.maximum(zn, 1)), -32767, 32767), NODATA_I16).astype(np.int16)
    return [p0, p1]


def encode_block(la: int, lo: int, planes: list[np.ndarray]) -> bytes:
    out = bytearray(b"TWF1") + struct.pack("<hhH", la, lo, len(planes))
    for p in planes:
        kind = 1 if p.dtype == np.int16 else 0
        if kind == 1:
            p = p.copy()
            p[:, 1:] = np.diff(p, axis=1)  # wraps in int16; decoding cumsums in int16 alike
        c = imagecodecs.zstd_encode(np.ascontiguousarray(p).tobytes(), level=19)
        out += struct.pack("<BHI", kind, p.shape[0], len(c)) + c
    return bytes(out)


def decode_block(data: bytes) -> tuple[int, int, list[np.ndarray]]:
    assert data[:4] == b"TWF1"
    la, lo, k = struct.unpack("<hhH", data[4:10])
    pos = 10
    planes = []
    for _ in range(k):
        kind, n, m = struct.unpack("<BHI", data[pos:pos + 7])
        raw = imagecodecs.zstd_decode(data[pos + 7:pos + 7 + m])
        pos += 7 + m
        p = np.frombuffer(raw, dtype=np.int16 if kind == 1 else np.uint8).reshape(n, n)
        if kind == 1:
            p = np.cumsum(p, axis=1, dtype=np.int16)
        planes.append(p)
    return la, lo, planes


def block_input(tiles: dict, la: int, lo: int) -> tuple[np.ndarray, np.ndarray | None]:
    """A block's window (MARGIN + 1 pixels round it) and GHSL gross height on it (None without the file)."""
    j0, i0 = block_window(la, lo)
    a = MARGIN + 1
    side = PX_PER_DEG + 2 * a
    z = window(tiles, j0 - a, i0 - a, side, side)
    b = built_reader()
    built = b.gross(*window_latlon(j0 - a, i0 - a, side, side)) if b is not None else None
    return z, built


def do_block(la: int, lo: int, paths: dict, outdir: str) -> tuple[int, int, str | None, int, float]:
    t = time.time()
    tiles = {k: decode_tile(Path(p).read_bytes()) for k, p in paths.items()}
    z, built = block_input(tiles, la, lo)
    del tiles
    if not np.isfinite(z).any():
        return la, lo, None, 0, time.time() - t
    data = encode_block(la, lo, summarise(z, la, built))
    name = block_name(la, lo)
    tmp = Path(outdir) / (name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, Path(outdir) / name)
    return la, lo, name, len(data), time.time() - t


# ---- the run

def run(blocks_path: str, outdir: str, jobs: int, fetchers: int, part_gb: float, send: str | None) -> None:
    import requests
    spec = json.loads(Path(blocks_path).read_text(encoding="utf-8"))
    url = spec["source"]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    tdir = out.parent / (out.name + "_tiles")
    tdir.mkdir(exist_ok=True)
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"source": url, "blocks": {}, "parts": []}
    todo = [b for b in spec["blocks"] if f"{b['la']}_{b['lo']}" not in manifest["blocks"]]
    t0 = time.time()
    fetch_built()
    if built_reader() is None:
        raise SystemExit(f"no GHSL built volume at {BUILT_PATH}: the tower filter needs it")
    print(f"GHSL built volume: {BUILT_PATH}", flush=True)
    print(f"{len(todo)} blocks to do of {len(spec['blocks'])}; reading the tile index", flush=True)
    offs, cnts = tile_index(url)
    across = -(-SRC_W // TILE)
    refs: dict = {}
    order = []
    for b in todo:
        for ty, tx in b["tiles"]:
            k = (ty, tx)
            if k not in refs:
                order.append(k)
                refs[k] = 0
            refs[k] += 1
    total_bytes = sum(int(cnts[ty * across + tx]) for ty, tx in order)
    print(f"{len(order)} tiles, {total_bytes / 1e9:.1f} GB to fetch", flush=True)

    have: dict = {}
    lock = threading.Lock()
    got = threading.Condition(lock)
    fetched = [0, 0]  # tiles, bytes
    q: "queue.Queue" = queue.Queue()
    ahead = threading.Semaphore(3000)  # tiles on disk at once; a band of blocks holds at most about 3 tile rows (1,900)
    for k in order:
        q.put(k)

    def fetcher():
        s = requests.Session()
        while True:
            try:
                k = q.get_nowait()
            except queue.Empty:
                return
            ahead.acquire()
            idx = k[0] * across + k[1]
            a, n = int(offs[idx]), int(cnts[idx])
            p = tdir / f"{k[0]}_{k[1]}.bin"
            for attempt in range(10):
                try:
                    r = s.get(url, headers={"Range": f"bytes={a}-{a + n - 1}"}, timeout=180)
                    if r.status_code == 206 and len(r.content) == n:
                        p.write_bytes(r.content)
                        break
                    print(f"tile {k}: HTTP {r.status_code}, {len(r.content)} of {n} bytes", flush=True)
                except Exception as e:
                    print(f"tile {k}: {type(e).__name__} {e}", flush=True)
                time.sleep(min(60, 2 ** attempt))
            else:
                p = None
            with got:
                have[k] = p
                fetched[0] += 1
                fetched[1] += n
                got.notify_all()

    threads = [threading.Thread(target=fetcher, daemon=True) for _ in range(fetchers)]
    for th in threads:
        th.start()

    pending = list(todo)
    part_bytes = 0
    part_files: list = []
    part_keys: dict = {}  # blocks done since the last part, written to the manifest when it ships
    last = 0.0
    done_n = 0
    failed = []

    def ship_part():
        nonlocal part_bytes, part_files, part_keys
        if not part_keys:
            return
        k = len(manifest["parts"])
        name = f"part{k}.tar"
        with tarfile.open(out / name, "w") as tf:
            for f in part_files:
                tf.add(out / f, arcname=f)
        for f in part_files:
            (out / f).unlink()
        manifest["blocks"].update(part_keys)
        manifest["parts"].append({"name": name, "files": len(part_files), "bytes": (out / name).stat().st_size,
                                  "blocks": sorted(part_keys)})
        man_path.write_text(json.dumps(manifest), encoding="utf-8")
        print(f"part {name}: {len(part_files)} blocks, {(out / name).stat().st_size / 1e9:.2f} GB", flush=True)
        if send:
            r = subprocess.run([send, str(out / name), str(k)])
            print(f"sent {name}: exit {r.returncode}", flush=True)
            if r.returncode == 0:
                (out / name).unlink()
        part_bytes, part_files, part_keys = 0, [], {}

    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futs = {}
        while pending or futs:
            with got:
                ready = [b for b in pending[:400] if all(tuple(t) in have for t in b["tiles"])][:max(0, jobs * 2 - len(futs))]
            for b in ready:
                pending.remove(b)
                paths = {tuple(t): str(have[tuple(t)]) for t in b["tiles"] if have[tuple(t)] is not None}
                if len(paths) < len(b["tiles"]):
                    failed.append(f"{b['la']}_{b['lo']}")
                futs[ex.submit(do_block, b["la"], b["lo"], paths, str(out))] = b
            if not futs:
                with got:
                    got.wait(timeout=5)
                continue
            fin, _ = wait(list(futs), timeout=5, return_when=FIRST_COMPLETED)
            for f in fin:
                b = futs.pop(f)
                la, lo, name, size, secs = f.result()
                key = f"{la}_{lo}"
                if key in failed:
                    name = None  # a tile could not be fetched: left out, so a rerun does it
                else:
                    part_keys[key] = name
                    if name:
                        part_files.append(name)
                        part_bytes += size
                done_n += 1
                for t in b["tiles"]:
                    t = tuple(t)
                    refs[t] -= 1
                    if refs[t] == 0:
                        p = have.get(t)
                        if p is not None and Path(p).exists():
                            Path(p).unlink()
                        ahead.release()
            if part_bytes >= part_gb * 1e9:
                ship_part()
            if time.time() - last > 60:
                last = time.time()
                el = time.time() - t0
                print(f"{time.strftime('%H:%M:%S')} blocks {done_n}/{len(todo)}, tiles {fetched[0]}/{len(order)} "
                      f"({fetched[1] / 1e9:.1f} GB, {fetched[1] / 1e6 / max(el, 1):.0f} MB/s), part {part_bytes / 1e9:.2f} GB, "
                      f"{el / 60:.0f} min", flush=True)
    ship_part()
    print(f"done: {done_n} blocks, {len([v for v in manifest['blocks'].values() if v])} files, "
          f"{len(failed)} left for a rerun {failed[:20]}, {(time.time() - t0) / 60:.0f} min", flush=True)
    man_path.write_text(json.dumps(manifest), encoding="utf-8")


def check(la: int, lo: int, path: str) -> None:
    """One block against rasterio's own read of the source: the decode and the cell mapping."""
    import rasterio
    from rasterio.windows import Window
    _, _, planes = decode_block(Path(path).read_bytes())
    j0, i0 = block_window(la, lo)
    a = MARGIN + 1
    side = PX_PER_DEG + 2 * a
    with rasterio.open("/vsicurl/" + SRC_URL) as d:
        z = d.read(1, window=Window(i0 - a, j0 - a, side, side), boundless=True, fill_value=np.nan).astype(np.float32)
        x, y = d.xy(j0, i0)  # centre of the block's first pixel
    print(f"first pixel centre {x:.6f}, {y:.6f}; block's north-west corner {lo}, {la + 1}")
    z = np.where((z > 1e30) | (z < -1000), np.nan, z)
    b = built_reader()
    ref = summarise(z, la, b.gross(*window_latlon(j0 - a, i0 - a, side, side)) if b else None)
    for k, (a, b) in enumerate(zip(planes, ref)):
        print(f"plane {k}: equal {np.array_equal(a, b)}, max diff {np.abs(a.astype(np.int32) - b.astype(np.int32)).max()}")


if __name__ == "__main__":
    a = sys.argv[1:]

    def opt(name, default):
        return type(default)(a[a.index(name) + 1]) if name in a else default

    if a and a[0] == "blocks":
        make_blocks(a[1], a[2:] or None)
    elif a and a[0] == "run":
        run(a[1], a[2], opt("--jobs", 4), opt("--fetch", 16), opt("--part-gb", 3.5), opt("--send", "") or None)
    elif a and a[0] == "check":
        check(int(a[1]), int(a[2]), a[3])
    else:
        print(__doc__)
