"""Publish the map: data/ to Cloudflare R2, index.html and the taxonomy to the website repo.

    python tools/deploy.py --dry-run     say what would move, touch nothing
    python tools/deploy.py               do it, then STOP before git

WHAT GOES WHERE, AND WHY IT IS ONE RULE RATHER THAN A JUDGEMENT PER FILE.  Everything under
`data/` goes to R2; `index.html` and `taxonomy/religions.json` go to the website repo.  The
archive is 84 MB and `data/buffers/` another 76 MB across 249 files, and both are rewritten by
every build, so committing them would add a fresh 160 MB to the website repo's history each
time.  The two files that do go in the repo are small and are the pair that decides what the
viewer can draw at all.

AND R2 RATHER THAN GITHUB PAGES FOR THE ARCHIVE IS NOT A SIZE DECISION.  PMTiles reads with
HTTP range requests, and Pages fronted by Cloudflare re-encodes the response and breaks them;
ancestrydots/COMMANDS.txt has the original diagnosis.  The symptom is the dangerous kind -- a
map that loads, with a basemap and a legend and no dots.  `--verify` checks it with a range
request afterwards, which must answer 206.

THIS SCRIPT NEVER COMMITS OR PUSHES.  It prints the git commands and stops.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEBSITE = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "website", "religiondots")

REMOTE = "r2:anita-maps/religiondots/data"
PUBLIC = "https://pub-ae551368cea941f39101e13c84d60bde.r2.dev/religiondots/data"

# The four things the viewer fetches out of data/.  `index.html`'s DATA_BASE builds every one
# of these paths, so a rename here is a rename there.
FILES = [
    "processed/religiondots.pmtiles",
    "processed/counts.json",
    "processed/country_shapes.geojson",
]
DIRS = ["buffers"]

# Copied into the website repo, keeping the same relative path.
REPO_FILES = ["index.html", "taxonomy/religions.json"]


def run(cmd, dry):
    print("   " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.call(cmd)


def preflight():
    """A half-finished build is the failure this catches, and it is invisible once uploaded.

    `build_tail.py` writes the archive, counts.json and the buffers in three steps under one
    lock.  If it was interrupted between them, every file exists and the set disagrees about
    which countries are drawn -- the viewer then greys out a country whose dots are sitting in
    the archive, or asks for a buffer that is not there and falls back to merged marks with no
    error.  Comparing the two country lists is the cheapest thing that notices.
    """
    problems = []
    for rel in FILES:
        p = os.path.join(ROOT, "data", rel)
        if not os.path.exists(p):
            problems.append(f"missing {rel} -- run tools/build_tail.py")
    man_p = os.path.join(ROOT, "data", "buffers", "manifest.json")
    if not os.path.exists(man_p):
        problems.append("missing data/buffers/manifest.json -- run buffers.py")
    if problems:
        return problems, None

    counts = json.load(open(os.path.join(ROOT, "data", "processed", "counts.json"),
                            encoding="utf-8"))
    man = json.load(open(man_p, encoding="utf-8"))
    drawn = set(counts.get("countries", {}))
    for ed_name, ed in man.get("editions", {}).items():
        have = set(ed.get("countries", {}))
        missing = sorted(drawn - have)
        extra = sorted(have - drawn)
        if missing:
            problems.append(f"buffers edition {ed_name} is missing {len(missing)} drawn "
                            f"countries: {', '.join(missing)} -- re-run buffers.py with ALL "
                            f"countries, --countries REPLACES the manifest")
        if extra:
            problems.append(f"buffers edition {ed_name} has {len(extra)} countries counts.json "
                            f"does not draw: {', '.join(extra)}")

    # Every .bin the manifest names has to be on disk, or the country silently loses its dots.
    for ed in man.get("editions", {}).values():
        for cc, ent in ed.get("countries", {}).items():
            f = ent.get("file") if isinstance(ent, dict) else None
            if f and not os.path.exists(os.path.join(ROOT, "data", "buffers", f)):
                problems.append(f"manifest names data/buffers/{f} and it is not on disk")
    return problems, len(drawn)


def verify():
    """A range request against the published archive.  206 or the dot layer is not there."""
    url = f"{PUBLIC}/processed/religiondots.pmtiles"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-99"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            code, n = r.status, len(r.read())
    except Exception as e:
        print(f"   FAILED {url}\n   {e}")
        return False
    ok = code == 206 and n == 100
    print(f"   {code} {n} bytes  {url}")
    if not ok:
        print("   NOT 206.  The archive is up but range requests are not being served, and the "
              "map will draw a basemap with no dots.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-data", action="store_true",
                    help="only refresh the website repo, when data/ has not changed")
    ap.add_argument("--prune", action="store_true",
                    help="rclone sync rather than copy for data/buffers, DELETING remote .bin "
                         "files no longer in the manifest.  Occasional cleanup, not routine.")
    ap.add_argument("--verify", action="store_true", help="range-check the archive and exit")
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify() else 1)

    problems, n = preflight()
    if problems:
        print("PREFLIGHT FAILED, nothing uploaded:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print(f"preflight ok: {n} countries, archive and buffers agree\n")

    if not a.skip_data:
        if not shutil.which("rclone"):
            print("rclone is not on PATH.  See COMMANDS.txt, DEPLOY.")
            sys.exit(1)
        print(f"data/ -> {REMOTE}")
        for rel in FILES:
            sub = os.path.dirname(rel)
            local = os.path.normpath(os.path.join(ROOT, "data", *rel.split("/")))
            rc = run(["rclone", "copy", "--checksum", "--progress",
                      local, f"{REMOTE}/{sub}/"], a.dry_run)
            if rc:
                print(f"   rclone exited {rc}.  A 403 here is the API token: see COMMANDS.txt.")
                sys.exit(rc)
        for d in DIRS:
            # copy, not sync: a stale .bin costs storage and nothing else, because the viewer
            # only ever asks for what the manifest names.  --prune when that adds up.
            verb = "sync" if a.prune else "copy"
            rc = run(["rclone", verb, "--checksum", "--progress",
                      os.path.normpath(os.path.join(ROOT, "data", d)),
                      f"{REMOTE}/{d}"], a.dry_run)
            if rc:
                sys.exit(rc)
        print()

    print(f"repo files -> {WEBSITE}")
    for rel in REPO_FILES:
        src = os.path.join(ROOT, rel.replace("/", os.sep))
        dst = os.path.join(WEBSITE, rel.replace("/", os.sep))
        print(f"   {rel}")
        if not a.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    if not a.dry_run and not a.skip_data:
        print("\nverifying the published archive")
        verify()

    print("\nNOT COMMITTED.  From the website repo:")
    print("   git add religiondots/")
    print('   git commit -m "religiondots"')
    print("   git push")
    print("\nThen: https://anita.garden/religiondots/")


if __name__ == "__main__":
    main()
