"""Remove old published versions of the game's data from R2 (r2:anitamaps/trainworld/<version>), safely.

    python trainworld/tools/r2_purge.py 2026-11-07c 2026-11-08          (dry run: says what it would do)
    python trainworld/tools/r2_purge.py 2026-11-07c 2026-11-08 --yes    (does it)

Refuses, whatever is asked:
- anything that is not a dated version (YYYY-MM-DD with an optional letter), so never terrain-world or the root;
- the version the live page reads (the tw-data meta in the website repo's trainsworld/index.html, and the one on
  https://anita.garden/trainsworld/ if it answers), so a purge can't break the game players have open;
- the KEEP newest versions on R2, so a tab opened on a recent hotfix keeps its data;
- a version not on R2 (a typo).
This is the one way to delete from R2; the permission rule allows this script, never a bare `rclone purge`.
"""
import argparse
import os
import re
import subprocess
import sys
import urllib.request

MAPS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REMOTE = "r2:anitamaps/trainworld"
SITE_PAGE = os.path.join(MAPS, "..", "website", "trainsworld", "index.html")
LIVE_URL = "https://anita.garden/trainsworld/"
KEEP = 3
DATED = re.compile(r"^\d{4}-\d{2}-\d{2}[a-z]?$")
META = re.compile(r'<meta name="tw-data" content="[^"]*/trainworld/([^/"]+)/"')


def live_versions():
    found = set()
    try:
        found.update(META.findall(open(SITE_PAGE, encoding="utf-8").read()))
    except OSError:
        pass
    try:
        req = urllib.request.Request(LIVE_URL, headers={"User-Agent": "r2-purge-check"})
        found.update(META.findall(urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")))
    except Exception as e:  # the repo copy alone still guards
        print(f"   (live page not read: {e})")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("versions", nargs="+")
    ap.add_argument("--yes", action="store_true", help="really purge (default: dry run)")
    a = ap.parse_args()
    out = subprocess.run(["rclone", "lsf", REMOTE + "/", "--dirs-only"], capture_output=True, text=True)
    if out.returncode:
        sys.exit("rclone lsf failed:\n" + out.stderr)
    on_r2 = sorted(d.rstrip("/") for d in out.stdout.split() if DATED.match(d.rstrip("/")))
    live = live_versions()
    if not live:
        sys.exit("could not find the live version (website trainsworld/index.html nor the live page): refusing")
    newest = set(on_r2[-KEEP:])
    print(f"on R2: {', '.join(on_r2)}\nlive: {', '.join(sorted(live))}\nkept as newest {KEEP}: {', '.join(sorted(newest))}")
    bad = []
    for v in a.versions:
        why = ("not a dated version" if not DATED.match(v) else "the live version" if v in live
               else f"one of the newest {KEEP}" if v in newest else "not on R2" if v not in on_r2 else None)
        if why:
            bad.append(f"{v}: {why}")
    if bad:
        sys.exit("refused, nothing purged:\n  " + "\n  ".join(bad))
    for v in a.versions:
        cmd = ["rclone", "purge", f"{REMOTE}/{v}", "--s3-no-check-bucket"]
        print(("" if a.yes else "(dry run) ") + " ".join(cmd))
        if a.yes and subprocess.run(cmd).returncode:
            sys.exit(f"purge of {v} failed")
    if not a.yes:
        print("dry run: add --yes to purge")


if __name__ == "__main__":
    main()
