"""The steps that rewrite whole-map files, under one lock: the language tree, then the tiles.

    python tools/build_tail.py --id <sid>     wait for the lock, then build everything drawn
    python tools/build_tail.py                say who holds the lock

tiles.py rebuilds ONE archive from every country's dots, so two agents running it at once both
write the same file. This takes an exclusive lock and derives the country list itself (every
countries/<cc>.py that loads, has a dots file AND is `drawn` in queue.csv), so nobody types the
list and drops one, and a country still being built cannot break the tail.
Waiting is the right outcome: the run you wait on very likely includes your country.

A few minutes with three countries; it grows with the map. Background it.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "build_tail.lock"
PROC = ROOT / "data" / "processed"
STALE = 3 * 3600

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def holder():
    try:
        return json.loads(LOCK.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id")
    ap.add_argument("--max-zoom", default="12")
    a = ap.parse_args()
    if not a.id:
        h = holder()
        print(f"held by {h['id']} since {time.strftime('%H:%M', time.localtime(h['since']))}" if h
              else "free")
        return
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({"id": a.id, "since": time.time(), "pid": os.getpid()}).encode())
            os.close(fd)
            break
        except FileExistsError:
            h = holder()
            if h and time.time() - h["since"] > STALE:
                print(f"lock held by {h['id']} for over 3 h: taking it over")
                LOCK.unlink(missing_ok=True)
                continue
            print(f"waiting for {h['id'] if h else '?'}…", flush=True)
            time.sleep(30)
    try:
        sys.path.insert(0, str(ROOT))
        from countries import load_all
        # ONLY COUNTRIES MARKED DRAWN (claim.py done), and the tree from only their fragments and
        # mappings: a country an agent is still building has half-written ones, and twice on
        # 2026-10-04 they stopped the whole tail
        with open(ROOT / "queue.csv", encoding="utf-8", newline="") as f:
            drawn = {r["cc"] for r in csv.DictReader(f) if r["status"] == "drawn"}
        ccs = sorted(cc for cc in load_all()
                     if cc in drawn and (PROC / f"dots_{cc}.geojson").exists())
        print(f"building {len(ccs)} countries: {','.join(ccs)}", flush=True)
        t0 = time.time()
        subprocess.run([sys.executable, str(ROOT / "taxonomy" / "build.py"), "--only", ",".join(ccs)],
                       check=True, cwd=ROOT, env={**os.environ, "LD_BUILD_TAIL": "1"})
        subprocess.run([sys.executable, str(ROOT / "tiles.py"), "--countries", ",".join(ccs),
                        "--max-zoom", a.max_zoom], check=True, cwd=ROOT)
        # the hatched not-drawn areas (2026-10-06); cached per country, about a minute
        subprocess.run([sys.executable, str(ROOT / "not_drawn.py")], check=True, cwd=ROOT)
        (PROC / "build_tail.json").write_text(json.dumps(
            {"finished": t0, "by": a.id, "countries": ccs}), encoding="utf-8")
        print(f"build tail done in {(time.time() - t0) / 60:.1f} min")
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
