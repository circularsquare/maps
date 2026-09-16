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

**UNDER A SUPERVISOR IT IS RUN IN BATCHES** (WORKFLOW_PLAN.md item 5). Builders stop after the
scatter; the supervisor runs this about hourly for every country waiting. A finished run writes
`data/build_last.json`, and `claim.py` lists the countries whose dots are newer than its start.
At the end it runs `coverage.py`, which reads the `counts.json` this run wrote, so only then does
the coverage check say anything about the new dots; a failure exits 1.

**A DEAD HOLDER'S LOCK IS CLEARED ON SIGHT.** The lock records its pid. If that process has gone,
or the pid now belongs to a process that started after the lock did, the lock is removed and the
wait carries on. Only a holder whose process cannot be checked still needs `--force` after
`STALE_MIN`; a live build is waited for however long it takes.

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


def _alive(rec):
    """Whether the lock holder's process is still running: True, False, or None if unknowable.

    Never ask with `os.kill(pid, 0)` on Windows, where it TERMINATES the process instead of
    probing it. And pids are reused, so a process that started after the lock was taken is
    somebody else's and the holder is gone."""
    pid = rec.get("pid")
    if not pid:
        return None
    if os.name != "nt":
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except OSError:
            return None
        return True
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x1000, False, int(pid))         # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        err = ctypes.get_last_error()
        return {5: True, 87: False}.get(err)             # access denied / no such process
    try:
        code = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
            return None
        if code.value != 259:                              # STILL_ACTIVE
            return False
        t = [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(h, *[ctypes.byref(x) for x in t]):
            return None
        created = ((t[0].dwHighDateTime << 32) | t[0].dwLowDateTime) / 1e7 - 11644473600
        return created <= rec.get("started", 0) + 5
    finally:
        k32.CloseHandle(h)


def _clear(judged):
    """Remove the lock only if it is still the one judged dead. Another waiter may have cleared
    it and taken a fresh one in between, and deleting that would cause the tear this prevents."""
    cur = _read() or {}
    if (cur.get("pid"), cur.get("started")) != (judged.get("pid"), judged.get("started")):
        return
    try:
        os.remove(LOCK)
    except FileNotFoundError:
        pass


def acquire(sid, wait=True, force=False):
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            cur = _read()
            if cur is None:
                # Empty or half-written: its holder may be writing it this instant.
                try:
                    fresh = time.time() - os.path.getmtime(LOCK) < 10
                except FileNotFoundError:
                    continue
                if fresh:
                    time.sleep(1)
                    continue
                cur = {}
            age = _age_min(cur)
            who = cur.get("id", "?")
            alive = _alive(cur)
            if force or alive is False:
                why = "--force" if force else f"its process {cur.get('pid')} is gone"
                print(f"!! clearing {who}'s lock ({age:.0f}m old) because {why}")
                _clear(cur)
                continue
            print(f"build lock held by {who}, {age:.0f}m ago, step: {cur.get('step', '?')}")
            if alive is None and age > STALE_MIN:
                print(f"   that is over {STALE_MIN}m and its process cannot be checked, so it may "
                      f"be a dead session.\n   Check nothing is running, then re-run with --force.")
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
    p.add_argument("--jobs", type=int, default=6,
                   help="tiles.py worker processes. 6 of Anita's 16 cores, because she uses "
                        "the machine while this runs; tiles are identical at any --jobs")
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

    started = time.time()
    py = sys.executable
    try:
        # Derived, never typed. --check first: it names a registered country missing an edition,
        # which is the one thing a hand-typed list cannot tell you.
        note("built_countries --check")
        subprocess.run([py, "tools/built_countries.py", "--check"], cwd=ROOT)

        got = subprocess.run([py, "tools/built_countries.py"], cwd=ROOT,
                             capture_output=True, text=True)
        # countries.py warns on stderr about a half-registered country and leaves it out of the
        # list. Captured here, so pass it on rather than build without saying so.
        if got.stderr.strip():
            print(got.stderr.rstrip())
        if got.returncode != 0:
            raise SystemExit(f"FAILED (exit {got.returncode}): tools/built_countries.py")
        ccs = got.stdout.strip()
        if not ccs:
            raise SystemExit("built_countries.py printed nothing — no country has both editions")
        print(f"\nbuilding {len(ccs.split(','))} countries")

        note("country_shapes")
        run([py, "country_shapes.py"])            # step 10, the one nothing depends on
        if args.rollup:
            note("rollup")
            run([py, "rollup.py"])
        note("tiles")
        run([py, "tiles.py", "--countries", ccs, "--coarse", "--no-atomic",
             "--jobs", str(args.jobs)])                                         # step 11
        note("buffers")
        run([py, "buffers.py", "--countries", ccs, "--coarse"])                # step 12
        # claim.py compares every country's dots with `started` to list what is still waiting.
        last = os.path.join(ROOT, "data", "build_last.json")
        with open(last + ".tmp", "w", encoding="utf-8") as fh:
            json.dump({"id": args.id, "started": started, "finished": time.time(),
                       "countries": ccs.split(",")}, fh, indent=1)
        os.replace(last + ".tmp", last)
    finally:
        release(args.id)

    # Only now does coverage.py say anything about this run's dots: it reads counts.json, which
    # tiles.py has just rewritten (COMMANDS.txt step 9).
    cov = subprocess.run([py, "coverage.py"], cwd=ROOT).returncode
    if cov:
        print("\n!! COVERAGE FAILED: a node that draws dots is missing from its country's "
              "coverage.\n   Fix coverage.py, then run the tail again, because tiles.py reads it.")

    print("\nbuild tail done. The one-line checks worth running now:")
    print("  python tools/built_countries.py --check")
    print("  python tools/check_tiles.py <ALL> --coarse --no-atomic --max-zoom 6")
    print("  python -c \"import json;m=json.load(open('data/buffers/manifest.json'));"
          "print({k:len(v['countries']) for k,v in m['editions'].items()})\"")
    return 1 if cov else 0


if __name__ == "__main__":
    sys.exit(main())
