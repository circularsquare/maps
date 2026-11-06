"""Copernicus GLO-90 tiles: names, the bucket's tile list, a streaming downloader with a disk guard,
and elevation on the fine grid (docs: notes/T-011.md until the world build lands).

    python isochrone/tools/terrain/glo90.py fetch <tile> [<tile> ...]   # keep them in data/raw/glo90/
    python isochrone/tools/terrain/glo90.py stream <list file> [--ahead GB] [--min-free GB] [--threads N]
                                                     # test run: download, read, delete, done list

Tiles are named by their south-west corner as the fine level's are (n46e023). Plain HTTPS GETs on
the public bucket (no account, no AWS tools). Every download first checks that C: keeps MIN_FREE
free after it; under that it sleeps and retries, logged. A file is written to .part and renamed
only once its length matches.
"""
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(HERE, "data", "raw", "glo90")
TERRAIN = os.path.join(HERE, "data", "terrain")
BUCKET = "https://copernicus-dem-90m.s3.amazonaws.com"
MIN_FREE = 22e9            # bytes C: must keep free after a download (Anita: leave ~20 GB)
AHEAD = 3e9                # bytes of downloaded, not yet used tiles at most
TILE_MAX = 40e6            # the largest tile, for the guard before its size is known
N = 1200                   # fine cells per tile side (1/1200 degree)
UA = "isochrone-terrain/1 (research map build)"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tile_name(lat, lon):
    return "%s%02d%s%03d" % ("n" if lat >= 0 else "s", abs(lat), "e" if lon >= 0 else "w", abs(lon))


def tile_latlon(name):
    lat = int(name[1:3]) * (1 if name[0] == "n" else -1)
    lon = int(name[4:7]) * (1 if name[3] == "e" else -1)
    return lat, lon


def cop_name(name):
    lat, lon = tile_latlon(name)
    return "Copernicus_DSM_COG_30_%s%02d_00_%s%03d_00_DEM" % (
        "N" if lat >= 0 else "S", abs(lat), "E" if lon >= 0 else "W", abs(lon))


def url(name):
    c = cop_name(name)
    return "%s/%s/%s.tif" % (BUCKET, c, c)


def raw_path(name, folder=RAW):
    return os.path.join(folder, name + ".tif")


def bucket_tiles():
    """Our names of every tile the bucket holds (tileList.txt, fetched once into data/raw/glo90/)."""
    p = os.path.join(RAW, "tileList.txt")
    if not os.path.exists(p):
        os.makedirs(RAW, exist_ok=True)
        fetch_url(BUCKET + "/tileList.txt", p)
    out = set()
    for line in open(p):
        s = line.strip()
        if not s.startswith("Copernicus_DSM_COG_30_"):
            continue
        p_ = s.split("_")             # Copernicus DSM COG 30 N46 00 E023 00 DEM
        lat = int(p_[4][1:]) * (1 if p_[4][0] == "N" else -1)
        lon = int(p_[6][1:]) * (1 if p_[6][0] == "E" else -1)
        out.add(tile_name(lat, lon))
    return out


def free_bytes(path=HERE):
    return shutil.disk_usage(path).free


_guard = {"lock": threading.Lock(), "waiting": 0, "last": 0.0}


def wait_for_space(need, min_free=MIN_FREE, poll=60):
    """Block until the drive keeps min_free after writing `need` bytes; logged on the first wait
    and every 10 minutes after, once for all threads."""
    waited = False
    while free_bytes() - need < min_free:
        with _guard["lock"]:
            if not waited:
                _guard["waiting"] += 1
                waited = True
            if time.time() - _guard["last"] > 600:
                _guard["last"] = time.time()
                log("disk guard: %.1f GB free, need %.1f GB + %.1f GB kept free; %d waiting"
                    % (free_bytes() / 1e9, need / 1e9, min_free / 1e9, _guard["waiting"]))
        time.sleep(poll)
    if waited:
        with _guard["lock"]:
            _guard["waiting"] -= 1
            if _guard["waiting"] == 0:
                _guard["last"] = 0.0
                log("disk guard: %.1f GB free, going on" % (free_bytes() / 1e9))


def fetch_url(u, path, tries=5):
    """GET u into path (via path.part); returns bytes written, or 0 when the bucket has no such key."""
    for t in range(tries):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                size = int(r.headers.get("Content-Length", 0))
                with open(path + ".part", "wb") as fh:
                    shutil.copyfileobj(r, fh, 1 << 20)
            got = os.path.getsize(path + ".part")
            if size and got != size:
                raise IOError("short read %d of %d" % (got, size))
            os.replace(path + ".part", path)
            return got
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return 0
            err = e
        except Exception as e:      # network: retry with backoff
            err = e
        log("retry %d %s: %s" % (t + 1, u, err))
        time.sleep(5 * 2 ** t)
    raise IOError("failed %s: %s" % (u, err))


def fetch(name, folder=RAW, min_free=MIN_FREE):
    """Download one tile unless present; returns its path, or None if the bucket has none (ocean)."""
    p = raw_path(name, folder)
    if os.path.exists(p):
        return p
    os.makedirs(folder, exist_ok=True)
    wait_for_space(TILE_MAX, min_free)
    return p if fetch_url(url(name), p) else None


FAILED = "failed"


class Stream:
    """Downloads tiles in list order on `threads` threads (one GET each), keeping at most `ahead`
    bytes downloaded and not yet released; iterate to get (name, path) in the same order (path None
    for a tile the bucket lacks, FAILED after every retry failed), then call release(name, path)
    once a tile is used, which deletes its raw file (unless keep)."""

    def __init__(self, names, folder=RAW, ahead=AHEAD, min_free=MIN_FREE, threads=6, keep=False):
        self.names, self.folder, self.ahead, self.min_free, self.keep = list(names), folder, ahead, min_free, keep
        self.threads = threads
        self.ready = {}
        self.held = 0                # bytes downloaded and not released
        self.flight = 0              # downloads under way
        self.nxt = 0                 # next list index to start
        self.want = 0                # list index the consumer waits for
        self.cv = threading.Condition()
        self.stop = False
        self.bytes = 0
        self.t0 = time.time()
        self.ths = [threading.Thread(target=self._run, daemon=True) for _ in range(threads)]
        for t in self.ths:
            t.start()

    def _run(self):
        while True:
            with self.cv:
                # past the buffer, wait, unless this is the tile the consumer needs next
                while not self.stop and self.nxt < len(self.names) and self.nxt > self.want and \
                        self.held + self.flight * TILE_MAX / 4 > self.ahead:
                    self.cv.wait(5)
                if self.stop or self.nxt >= len(self.names):
                    return
                i = self.nxt
                self.nxt += 1
                self.flight += 1
            name = self.names[i]
            try:
                wait_for_space(TILE_MAX * self.threads, self.min_free)
                p = fetch(name, self.folder, self.min_free)
            except Exception as e:
                log("download failed", name, e)
                p = FAILED
            sz = os.path.getsize(p) if p and p != FAILED else 0
            with self.cv:
                self.flight -= 1
                self.ready[name] = p
                self.held += sz
                self.bytes += sz
                self.cv.notify_all()

    def __iter__(self):
        for i, name in enumerate(self.names):
            with self.cv:
                self.want = i
                self.cv.notify_all()
                while name not in self.ready:
                    self.cv.wait(5)
                p = self.ready.pop(name)
            yield name, p

    def release(self, name, path):
        if path and path != FAILED and os.path.exists(path):
            sz = os.path.getsize(path)
            if not self.keep:
                os.remove(path)
            with self.cv:
                self.held -= sz
                self.cv.notify_all()

    def close(self):
        with self.cv:
            self.stop = True
            self.cv.notify_all()

    def rate(self):
        return self.bytes / max(1e-9, time.time() - self.t0)


# ---------------------------------------------------------------- elevation on the fine grid

def read_fine(path, name):
    """Elevation (float32 metres) at the centres of the tile's 1200 x 1200 fine cells, bilinear from
    the GLO-90 posts. GLO-90 is pixel-is-point: its posts sit on multiples of 1/1200 degree in
    latitude (the fine grid's cell corners), and of 1/1200 x 1.5, 2, 3, 4 degree in longitude north
    of 50, 60, 70, 80 degrees. Missing edge posts (the tile holds no south row or east column) repeat
    the last one. Ocean in a coastal tile is 0 m in GLO-90."""
    import rasterio
    lat, lon = tile_latlon(name)
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        t = src.transform
        nod = src.nodata
    if nod is not None:
        a[a == nod] = 0
    h, w = a.shape
    # post (i, j) is at lon = t.c + (j + 0.5) t.a, lat = t.f + (i + 0.5) t.e (rasterio's area convention)
    x = lon + (np.arange(N) + 0.5) / N
    y = lat + 1 - (np.arange(N) + 0.5) / N
    fx = (x - t.c) / t.a - 0.5
    fy = (y - t.f) / t.e - 0.5
    j0 = np.clip(np.floor(fx).astype(int), 0, w - 1)
    i0 = np.clip(np.floor(fy).astype(int), 0, h - 1)
    j1, i1 = np.minimum(j0 + 1, w - 1), np.minimum(i0 + 1, h - 1)
    wx = np.clip(fx - j0, 0, 1).astype(np.float32)[None, :]
    wy = np.clip(fy - i0, 0, 1).astype(np.float32)[:, None]
    top = a[np.ix_(i0, j0)] * (1 - wx) + a[np.ix_(i0, j1)] * wx
    bot = a[np.ix_(i1, j0)] * (1 - wx) + a[np.ix_(i1, j1)] * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


# ---------------------------------------------------------------- command line

def main(argv):
    if argv[0] == "fetch":
        have = bucket_tiles()
        for name in argv[1:]:
            if name not in have:
                log(name, "not in the bucket (ocean)")
                continue
            t0 = time.time()
            p = fetch(name)
            log(name, "%.1f MB" % (os.path.getsize(p) / 1e6), "%.1f s" % (time.time() - t0),
                "free %.1f GB" % (free_bytes() / 1e9))
    elif argv[0] == "stream":
        # a test of the streaming loop: download ahead, read each tile to the fine grid, delete it,
        # record it in a done list; a restart skips what the list has
        names = [s.strip() for s in open(argv[1]) if s.strip()]
        ahead = float(argv[argv.index("--ahead") + 1]) * 1e9 if "--ahead" in argv else AHEAD
        mf = float(argv[argv.index("--min-free") + 1]) * 1e9 if "--min-free" in argv else MIN_FREE
        folder = os.path.join(RAW, "stream_test")
        done_p = os.path.join(TERRAIN, "stream_test_done.txt")
        os.makedirs(TERRAIN, exist_ok=True)
        done = set(s.strip() for s in open(done_p)) if os.path.exists(done_p) else set()
        have = bucket_tiles()
        todo = [n for n in names if n not in done and n in have]
        log("stream test: %d tiles, %d done before, ahead %.1f GB, min free %.1f GB"
            % (len(todo), len(done), ahead / 1e9, mf / 1e9))
        threads = int(argv[argv.index("--threads") + 1]) if "--threads" in argv else 6
        st = Stream(todo, folder, ahead, mf, threads)
        t0 = time.time()
        try:
            for k, (name, p) in enumerate(st):
                if p == FAILED:
                    log(name, "failed; left for a rerun")
                    continue
                z = read_fine(p, name) if p else None
                st.release(name, p)
                with open(done_p, "a") as fh:
                    fh.write(name + "\n")
                log("%d/%d %s max %.0f m, %.1f MB/s, held %.2f GB, free %.1f GB"
                    % (k + 1, len(todo), name, z.max() if z is not None else -1, st.rate() / 1e6,
                       st.held / 1e9, free_bytes() / 1e9))
        finally:
            st.close()
        log("done in %.0f s" % (time.time() - t0))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
