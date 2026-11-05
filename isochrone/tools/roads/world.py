"""The world roads pass (T-053; docs/roads.md "Fetching the world").

    python isochrone/tools/roads/world.py [--jobs N] [--only id,id,...] [--min-free GB]

Goes down data/roads/regions.json: downloads each extract (tools/roads/fetch.py, md5 checked),
reads it with `build.py extract` in a process of its own, and deletes it once read. Never more
than two extracts on disk (downloading, waiting or being read); the next one downloads while
one is read. --jobs extracts are read at once (1 or 2).

Extracts listed in data/roads/done.txt are skipped, so a rerun carries on where a stop left off.
Before each download the free space on the drive is checked: if it would fall under MIN_FREE,
no more are started, the ones being read finish, and the run stops (exit code 3). A failed
download (3 tries) or read is logged in data/roads/failed.txt and the run goes on.
"""
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import fetch  # noqa: E402

ROADS = os.path.join(HERE, "data", "roads")
REGIONS = os.path.join(ROADS, "regions.json")
DONE = os.path.join(ROADS, "done.txt")
FAILED = os.path.join(ROADS, "failed.txt")
OSM = fetch.OSM
MIN_FREE = 6e9          # bytes left on the drive after a download (Anita, 2026-11-05)
SLOTS = 2               # extracts on disk at once


def log(msg):
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def name_of(r):
    return r["id"].replace("/", "_")      # ids are unique; last parts are not (us/georgia, georgia)


def region_of(r):
    return r["url"][len(fetch.BASE):-len("-latest.osm.pbf")]


def read_list(path):
    return set(open(path).read().split()) if os.path.exists(path) else set()


def main():
    args = sys.argv[1:]
    jobs = 1
    only = None
    if "--jobs" in args:
        jobs = max(1, min(SLOTS, int(args[args.index("--jobs") + 1])))
    global MIN_FREE
    if "--min-free" in args:
        MIN_FREE = float(args[args.index("--min-free") + 1]) * 1e9
    if "--only" in args:
        only = args[args.index("--only") + 1].split(",")
    regions = json.load(open(REGIONS))
    if only:
        regions = [r for r in regions if r["id"] in only]
    done = read_list(DONE)
    queue = [r for r in regions if name_of(r) not in done]
    total = sum(r["bytes"] for r in queue)
    log("%d extracts to go (%.1f GB), %d done already, %d at a time" % (len(queue), total / 1e9, len(regions) - len(queue), jobs))
    ready, reading = [], {}          # downloaded names waiting; name -> (process, start, bytes)
    dl = {"thread": None, "name": None, "ok": None}
    stopping = False
    t0 = time.time()
    got = 0

    def download(r, name):
        for attempt in range(3):
            try:
                fetch.fetch(region_of(r), name)
                dl["ok"] = True
                return
            except Exception as e:  # noqa: BLE001
                log("download %s failed (%s), try %d" % (name, e, attempt + 1))
                time.sleep(30)
        dl["ok"] = False

    byname = {name_of(r): r for r in regions}
    while queue or ready or reading or dl["thread"]:
        # a download that has ended
        if dl["thread"] and not dl["thread"].is_alive():
            if dl["ok"]:
                ready.append(dl["name"])
            else:
                with open(FAILED, "a") as f:
                    f.write("%s download\n" % dl["name"])
                part = os.path.join(OSM, dl["name"] + ".osm.pbf.part")
                if os.path.exists(part):
                    os.remove(part)
            dl["thread"] = None
        # reads that have ended
        for name, (p, ts, nb) in list(reading.items()):
            if p.poll() is None:
                continue
            del reading[name]
            path = os.path.join(OSM, name + ".osm.pbf")
            if os.path.exists(path):
                os.remove(path)
            if p.returncode == 0:
                got += nb
                el = time.time() - t0
                rate = got / el if el > 0 else 0
                left = total - got
                log("done %s (%.0f MB, %.0f s); %.1f / %.1f GB, ~%.1f h to go" % (
                    name, nb / 1e6, time.time() - ts, got / 1e9, total / 1e9, left / rate / 3600 if rate else 0))
            else:
                log("read %s failed (exit %d), see data/roads/logs/%s.log" % (name, p.returncode, name))
                with open(FAILED, "a") as f:
                    f.write("%s read\n" % name)
        # start a download when a slot is free
        on_disk = len(ready) + len(reading) + (1 if dl["thread"] else 0)
        if not stopping and queue and not dl["thread"] and on_disk < SLOTS:
            r = queue[0]
            free = shutil.disk_usage(OSM).free
            if free - r["bytes"] < MIN_FREE:
                stopping = True
                log("stopping: %.1f GB free, %s needs %.2f GB, keeping %.0f GB" % (free / 1e9, r["id"], r["bytes"] / 1e9, MIN_FREE / 1e9))
            else:
                queue.pop(0)
                dl.update(name=name_of(r), ok=None)
                dl["thread"] = threading.Thread(target=download, args=(r, name_of(r)), daemon=True)
                dl["thread"].start()
                log("download %s (%.0f MB, %.1f GB free)" % (r["id"], r["bytes"] / 1e6, free / 1e9))
        if stopping:
            queue = []
        # start reads
        while ready and len(reading) < jobs:
            name = ready.pop(0)
            os.makedirs(os.path.join(ROADS, "logs"), exist_ok=True)
            lf = open(os.path.join(ROADS, "logs", name + ".log"), "w")
            p = subprocess.Popen([sys.executable, os.path.join(TOOLS, "build.py"), "extract", name],
                                 stdout=lf, stderr=subprocess.STDOUT)
            reading[name] = (p, time.time(), byname[name]["bytes"])
            log("read %s (pid %d)" % (name, p.pid))
        time.sleep(2)
    if stopping:
        log("stopped for disk space; free some and run again to carry on")
        sys.exit(3)
    log("all extracts done in %.1f h; %d failed (data/roads/failed.txt)" % ((time.time() - t0) / 3600, len(open(FAILED).read().splitlines()) if os.path.exists(FAILED) else 0))


if __name__ == "__main__":
    main()
