"""The terrain run (T-011 stage 2): stream GLO-90 over the plan's tiles and keep only what is derived.

    python isochrone/tools/terrain/world.py run [--workers N] [--threads 6] [--ahead 2] [--min-free 22]
                                               [--tiles a,b,...] [--out DIR]
    python isochrone/tools/terrain/world.py status [--out DIR]

Order: data/terrain/plan.json (tools/terrain/plan.py), north to south, west to east. Per tile:
  1. download (glo90.Stream: `threads` GETs at once, at most `ahead` GB not yet used, never under
     `min-free` GB free on the drive),
  2. elevation at fine centres (glo90.read_fine) to <out>/work/<tile>.npy (decimetres, int32), the
     raw tile deleted; tiles the fine level will want keep theirs in <out>/elev/<tile>.bin.gz
     (int16 metres, column deltas, blocks outside a town tile's mask zeroed),
  3. once the tile and its eight neighbours have elevation (or have none: no GLO-90 tile, or not in
     the plan), its L1 crossings (walk.tile_l1, on `workers` processes) to <out>/l1/<tile>.bin.gz,
     the tile in <out>/done.txt; a work file goes when every plan tile next to it is done.
A restart skips done tiles and fetches again only what a tile still to do needs. Log: <out>/world.log;
<out>/world.pid holds the running process id; `status` prints both.
"""
import gzip
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glo90 as G          # noqa: E402
import walk as Wk          # noqa: E402

N = 1200
HALO = Wk.HALO


def arg(a, k, default, f=str):
    return f(a[a.index(k) + 1]) if k in a else default


def neighbours(name):
    lat, lon = G.tile_latlon(name)
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                out.append(((di, dj), G.tile_name(lat - di, ((lon + dj + 180) % 360) - 180)))
    return out


def keep_bytes(z, mask):
    m = np.rint(z).astype(np.int16)
    if mask != 0xFFFF:
        b = ((mask >> np.arange(16)) & 1).astype(bool).reshape(4, 4)
        m[~np.repeat(np.repeat(b, 300, 0), 300, 1)] = 0
    d = np.diff(m.astype(np.int32), axis=1, prepend=0).astype("<i2")
    return gzip.compress(d.tobytes(), 9)


def write_atomic(path, data):
    with open(path + ".tmp", "wb") as fh:
        fh.write(data)
    os.replace(path + ".tmp", path)


def compute(job):
    """One tile's L1 crossings; job = (name, out dir, neighbour names with work files)."""
    name, out, nb = job
    t0 = time.time()
    work = os.path.join(out, "work")
    z = np.load(os.path.join(work, name + ".npy")).astype(np.float32) / 10
    znb = {}
    for (di, dj), nm in neighbours(name):
        if nm in nb:
            znb[(di, dj)] = np.load(os.path.join(work, nm + ".npy")).astype(np.float32) / 10
    lat, lon = G.tile_latlon(name)
    res = Wk.tile_l1(name, z, znb)
    write_atomic(os.path.join(out, "l1", name + ".bin.gz"), gzip.compress(res["bytes"], 6))
    return name, time.time() - t0, res["stats"]


def status(out):
    p = os.path.join(out, "world.pid")
    pid = open(p).read().strip() if os.path.exists(p) else None
    alive = False
    if pid:
        r = subprocess.run(["tasklist", "/FI", "PID eq %s" % pid], capture_output=True, text=True)
        alive = pid in r.stdout
    done = sum(1 for _ in open(os.path.join(out, "done.txt"))) if os.path.exists(os.path.join(out, "done.txt")) else 0
    plan = json.load(open(os.path.join(G.TERRAIN, "plan.json")))["tiles"]
    lp = os.path.join(out, "world.log")
    age = time.time() - os.path.getmtime(lp) if os.path.exists(lp) else None
    print("pid %s %s; done %d of %d; log last written %s s ago; free %.1f GB"
          % (pid, "running" if alive else "not running", done, len(plan), None if age is None else int(age),
             G.free_bytes() / 1e9))
    if os.path.exists(lp):
        print("".join(open(lp).readlines()[-5:]), end="")


def run(a):
    out = arg(a, "--out", G.TERRAIN)
    for sub in ("work", "l1", "elev"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    logf = open(os.path.join(out, "world.log"), "a")

    def log(*x):
        s = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(v) for v in x)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    write_atomic(os.path.join(out, "world.pid"), str(os.getpid()).encode())
    plan = json.load(open(os.path.join(G.TERRAIN, "plan.json")))["tiles"]
    if "--tiles" in a:
        want = set(arg(a, "--tiles", "").split(","))
        plan = [t for t in plan if t["name"] in want]
    info = {t["name"]: t for t in plan}
    order = [t["name"] for t in plan]
    bucket = G.bucket_tiles()
    done_p = os.path.join(out, "done.txt")
    done = set(s.strip() for s in open(done_p)) if os.path.exists(done_p) else set()
    work = os.path.join(out, "work")
    have = set(f[:-4] for f in os.listdir(work) if f.endswith(".npy"))
    todo = [n for n in order if n not in done]
    # what has to be fetched: tiles to do and their plan neighbours, unless a work file is there
    need = set(todo)
    for n in todo:
        need |= {nm for _, nm in neighbours(n) if nm in info}
    fetch = [n for n in order if n in need and n not in have and n in bucket]
    workers = arg(a, "--workers", 3, int)
    log("run: %d tiles, %d done, %d to fetch, %d workers, free %.1f GB"
        % (len(order), len(done), len(fetch), workers, G.free_bytes() / 1e9))

    def ready(n):
        """Elevation of the tile and of every neighbour that can have some."""
        if n not in have:
            return False
        return all(nm in have or nm not in info or nm not in bucket for _, nm in neighbours(n))

    def nb_have(n):
        return {nm for _, nm in neighbours(n) if nm in have}

    pending = [n for n in todo]
    started = set()
    t0 = time.time()
    ndone = 0
    st = G.Stream(fetch, G.RAW, arg(a, "--ahead", 2.0, float) * 1e9, arg(a, "--min-free", 22.0, float) * 1e9,
                  arg(a, "--threads", 6, int))
    it = iter(st)
    pool = Pool(workers)
    inflight = {}

    def submit_ready():
        for n in pending:
            if n not in started and ready(n) and len(inflight) < 2 * workers:
                started.add(n)
                inflight[n] = pool.apply_async(compute, ((n, out, nb_have(n)),))

    def collect(block):
        nonlocal ndone
        for n in list(inflight):
            r = inflight[n]
            if block or r.ready():
                name, sec, stats = r.get()
                del inflight[n]
                with open(done_p, "a") as fh:
                    fh.write(name + "\n")
                done.add(name)
                ndone += 1
                log("done %s %.1f s %s (%d of %d this run, %.0f s, %.1f MB/s, free %.1f GB)"
                    % (name, sec, json.dumps(stats), ndone, len(todo), time.time() - t0, st.rate() / 1e6,
                       G.free_bytes() / 1e9))
                # drop work files no tile still needs
                for m in [name] + [nm for _, nm in neighbours(name)]:
                    if m in have and all(x in done or x not in info for x in [m] + [nm for _, nm in neighbours(m)]):
                        p = os.path.join(work, m + ".npy")
                        if os.path.exists(p):
                            os.remove(p)
                        have.discard(m)
                if block:
                    return

    try:
        exhausted = False
        while len(done & set(todo)) < len(todo):
            if not exhausted and len(inflight) < 2 * workers:
                try:
                    name, p = next(it)
                except StopIteration:
                    exhausted = True
                    name = None
                if name:
                    if p == G.FAILED:
                        log("failed %s: left for a rerun" % name)
                    elif p:
                        z = G.read_fine(p, name)
                        np.save(os.path.join(work, name + ".npy.tmp.npy"), np.rint(z * 10).astype(np.int32))
                        os.replace(os.path.join(work, name + ".npy.tmp.npy"), os.path.join(work, name + ".npy"))
                        if info.get(name, {}).get("keep"):
                            write_atomic(os.path.join(out, "elev", name + ".bin.gz"), keep_bytes(z, info[name]["keep"]))
                        have.add(name)
                    st.release(name, p)
            submit_ready()
            collect(False)
            if exhausted:
                if not inflight:
                    left = [n for n in todo if n not in done]
                    if left:
                        log("stopping: %d tiles cannot run (a neighbour failed): %s" % (len(left), ",".join(left[:20])))
                    break
                collect(True)
            elif len(inflight) >= 2 * workers:
                collect(True)
        log("run ends: %d done this run in %.0f s" % (ndone, time.time() - t0))
    finally:
        st.close()
        pool.close()
        pool.join()


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "run":
        run(a[1:])
    elif a[0] == "status":
        status(arg(a, "--out", G.TERRAIN))
    else:
        sys.exit(__doc__)
