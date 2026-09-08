"""Run the whole-map build tail under an exclusive lock. COMMANDS.txt steps 10-12, serialised.

    python tools/build_tail.py --id <sid>            # wait for the lock, then build
    python tools/build_tail.py --id <sid> --no-wait  # exit 2 rather than wait
    python tools/build_tail.py                       # who holds the lock right now

**THE COLLISION THIS EXISTS FOR.** `claim.py` locks a COUNTRY, which stops two sessions building
Peru twice. It does nothing about the last three steps, and those are not per-country at all:
`country_shapes.py` rewrites one geojson for every country, `tiles.py --countries` REPLACES the
whole archive, and `buffers.py --countries` REPLACES the whole manifest. Two agents finishing
within twenty minutes of each other both run all three over the same files, and step 11 is
documented to run `--no-atomic`, which writes the archive in place rather than swapping it. So
the loser does not lose a merge, it reads a `.pmtiles` that another process is mid-write.

Nothing detected this before, because the symptom is not an error. It is a handful of wrong
tiles, or a manifest missing an edition, on a map that looks built. `sources.md` already carries
the near-miss version of this (six bad tiles all tagged `c='in'`, diagnosed from three mtimes).

**IT ALSO CLOSES "THE THREE SILENT ONES".** The country list is derived and never typed, `--coarse`
is on both commands, and `country_shapes.py` cannot be forgotten, because all of that is here
rather than in three commands somebody pastes.

**WAITING IS CORRECT AND COSTS NOTHING.** These steps rebuild EVERY country from whatever dots are
on disk, so a run you waited for probably already included your country, and the run you then do
yourself is merely redundant rather than wrong. Twenty minutes of waiting is much cheaper than
the hour of diagnosis a torn archive costs.

Not in scope, because neither is whole-map-destructive and both are cheap:
`taxonomy/build_tree.py` (run it yourself after touching a mapping, before scattering) and
`rollup.py` (after editing a COLUMNS dict; `--rollup` here runs it inside the lock if you want).
"""

import argparse
import json
import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOCK = os.path.join(ROOT, "data", "build.lock")

# A full tiles.py + buffers.py over every country is ~15-25 min and grows with the map, so this
# is deliberately generous. Past it the holder is reported as probably dead, and is still not
# broken automatically -- stealing a live build lock would cause the exact tear it prevents.
STALE_MIN = 60
POLL_SEC = 30


def _read():
    try:
        with open(LOCK, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _age_min(rec):
    return (time.time() - rec.get("started", 0)) / 60.0


def acquire(sid, wait=True, force=False):
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            cur = _read() or {}
            age = _age_min(cur)
            who = cur.get("id", "?")
            if force:
                print(f"!! breaking {who}'s lock ({age:.0f}m old) because --force")
                os.remove(LOCK)
                continue
            print(f"build lock held by {who}, {age:.0f}m ago, step: {cur.get('step', '?')}")
            if age > STALE_MIN:
                print(f"   that is over {STALE_MIN}m, so it is probably a dead session.\n"
                      f"   Check nothing is running, then re-run with --force.")
                return False
            if not wait:
                return False
            print(f"   waiting {POLL_SEC}s. This is fine: the build covers every country, so "
                  f"their run\n   very likely includes yours already.")
            time.sleep(POLL_SEC)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"id": sid, "started": time.time(), "pid": os.getpid(),
                       "step": "starting"}, fh, indent=1)
        return True


def note(step):
    rec = _read() or {}
    rec["step"] = step
    tmp = LOCK + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    os.replace(tmp, LOCK)


def release(sid):
    cur = _read()
    if cur and cur.get("id") != sid:
        print(f"!! not releasing: the lock now belongs to {cur.get('id')}, not to you")
        return
    try:
        os.remove(LOCK)
    except OSError:
        pass


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"FAILED (exit {r.returncode}): {' '.join(cmd)}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--id", help="your session id; omit to just report the lock")
    p.add_argument("--no-wait", action="store_true", help="exit 2 rather than queue")
    p.add_argument("--force", action="store_true",
                   help="break a lock you are sure is dead; check for a running python first")
    p.add_argument("--rollup", action="store_true",
                   help="also rebuild rollup.json, after editing a COLUMNS dict")
    args = p.parse_args()

    if not args.id:
        cur = _read()
        if not cur:
            print("build lock is free")
        else:
            print(f"held by {cur.get('id')}, {_age_min(cur):.0f}m ago, "
                  f"step: {cur.get('step', '?')}, pid {cur.get('pid')}")
        return 0

    if not acquire(args.id, wait=not args.no_wait, force=args.force):
        print("\nnot building. Your dots are on disk, so the next tail run picks the country up;\n"
              "say so in your final message and stop, or re-run this later.")
        return 2

    py = sys.executable
    try:
        # Derived, never typed. --check first: it names a registered country missing an edition,
        # which is the one thing a hand-typed list cannot tell you.
        note("built_countries --check")
        subprocess.run([py, "tools/built_countries.py", "--check"], cwd=ROOT)

        ccs = subprocess.run([py, "tools/built_countries.py"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        if not ccs:
            raise SystemExit("built_countries.py printed nothing — no country has both editions")
        print(f"\nbuilding {len(ccs.split(','))} countries")

        note("country_shapes")
        run([py, "country_shapes.py"])            # step 10, the one nothing depends on
        if args.rollup:
            note("rollup")
            run([py, "rollup.py"])
        note("tiles")
        run([py, "tiles.py", "--countries", ccs, "--coarse", "--no-atomic"])   # step 11
        note("buffers")
        run([py, "buffers.py", "--countries", ccs, "--coarse"])                # step 12
    finally:
        release(args.id)

    print("\nbuild tail done. The one-line checks worth running now:")
    print("  python tools/built_countries.py --check")
    print("  python tools/check_tiles.py <ALL> --coarse --no-atomic --max-zoom 6")
    print("  python -c \"import json;m=json.load(open('data/buffers/manifest.json'));"
          "print({k:len(v['countries']) for k,v in m['editions'].items()})\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
