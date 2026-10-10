"""Publish the map: the data to Cloudflare R2, index.html to the website repo.

    python tools/deploy.py --dry-run     preflight, then say what would move; touch nothing
    python tools/deploy.py               preflight, upload, copy index.html, verify, STOP before git
    python tools/deploy.py --verify      range-check the published archive on its own

religiondots/tools/deploy.py is the model and its docstrings have the long reasoning; this is the
same thing for one archive and four small files.

WHAT GOES WHERE: everything the page fetches goes to R2 under `r2:anitamaps/languagedots/`, at
the SAME relative path it has here (data/processed/..., taxonomy/languages.json,
country_shapes.geojson), and only index.html goes to the website repo. The page's dataBaseP
decides between the two by asking for data/processed/build_tail.json, which exists only in this
tree and is never uploaded. religiondots also puts its taxonomy JSON in the repo; here it goes to
R2 with the rest, because it is 0.65 MB and rewritten by every build tail, and one rule is
simpler than two.

THE ARCHIVE GOES UP RAW, THE REST PRE-GZIPPED. R2 compresses nothing on its own, so the JSON
and GeoJSON are uploaded gzipped with `Content-Encoding: gzip` and browsers inflate them
transparently. The archive must stay raw: PMTiles reads it with byte ranges, and a range into
a gzip stream is not a range into the file. Its tiles are gzipped inside it anyway.

THIS SCRIPT NEVER COMMITS OR PUSHES. It prints the git commands and stops.
"""

import argparse
import csv
import gzip
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                       # maps/languagedots
MAPS = os.path.dirname(ROOT)
WEBSITE = os.path.join(os.path.dirname(MAPS), "website", "languagedots")   # projects/website/<map>
PROC = os.path.join(ROOT, "data", "processed")
LOCK = os.path.join(ROOT, "build_tail.lock")

REMOTE = "r2:anitamaps/languagedots"
PUBLIC = "https://pub-ae551368cea941f39101e13c84d60bde.r2.dev/languagedots"

# Paths relative to ROOT, and the same paths under REMOTE. index.html builds each of these from
# DATA_BASE, so a rename here is a rename there.
RAW_FILES = ["data/processed/languagedots.pmtiles"]
GZ_FILES = ["data/processed/counts.json", "data/processed/not_drawn.geojson",
            "taxonomy/languages.json", "country_shapes.geojson", "admin1_shapes.geojson"]
REPO_FILES = ["index.html"]

# Under data/, which is gitignored, so the compressed copies never reach a repo.
STAGE = os.path.join(ROOT, "data", "gz_stage")

# --s3-no-check-bucket: without it rclone calls CreateBucket first, an admin operation these
# Object Read & Write tokens are refused, and the copy dies with a 403 that reads as a dead token.
RCLONE = ["rclone", "--s3-no-check-bucket", "--checksum", "--progress"]
GZ_HEADER = ["--header-upload", "Content-Encoding: gzip"]

# r2.dev is behind Cloudflare's bot rules; curl's own agent passes, a bare Python one does not.
CURL = ["curl", "-s", "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/141.0"]


def mtime(rel):
    return os.path.getmtime(os.path.join(ROOT, *rel.split("/")))


def hhmm(t):
    return time.strftime("%m-%d %H:%M", time.localtime(t))


def preflight():
    """Catch a half-finished or stale build before it is uploaded, where it is invisible."""
    problems = []

    # 1. nobody is in the build tail: it rewrites the archive, counts.json and not_drawn in turn
    if os.path.exists(LOCK):
        try:
            h = json.load(open(LOCK, encoding="utf-8"))
            who = f"{h.get('id')} since {hhmm(h.get('since', 0))} (pid {h.get('pid')})"
        except (OSError, ValueError):
            who = "unknown holder"
        problems.append(f"build tail running: build_tail.lock held by {who}. Wait for it.")

    # 2. every file present
    for rel in RAW_FILES + GZ_FILES + ["data/processed/build_tail.json"]:
        if not os.path.exists(os.path.join(ROOT, *rel.split("/"))):
            problems.append(f"missing {rel} -- run tools/build_tail.py")
    if problems:
        return problems, None

    # 3. counts.json parses, and agrees with the last tail about what is drawn
    try:
        counts = json.load(open(os.path.join(PROC, "counts.json"), encoding="utf-8"))
    except ValueError as e:
        return [f"counts.json does not parse: {e}"], None
    drawn = set(counts.get("countries", {}))
    tail = json.load(open(os.path.join(PROC, "build_tail.json"), encoding="utf-8"))
    built = set(tail.get("countries", []))
    if drawn != built:
        problems.append(f"counts.json and build_tail.json disagree: only in counts "
                        f"{sorted(drawn - built)}, only in the tail's list {sorted(built - drawn)}")

    # 4. nothing waiting for the tail (claim.py's own test: dots newer than the last tail start)
    waiting = sorted(f[5:-8] for f in os.listdir(PROC)
                     if f.startswith("dots_") and f.endswith(".geojson") and "_" not in f[5:-8]
                     and os.path.getmtime(os.path.join(PROC, f)) > tail.get("finished", 0))
    if waiting:
        problems.append(f"waiting for the build tail (claim.py): {' '.join(waiting)}")
    with open(os.path.join(ROOT, "queue.csv"), encoding="utf-8", newline="") as f:
        queued = {r["cc"] for r in csv.DictReader(f) if r["status"] == "drawn"}
    if queued - drawn:
        problems.append(f"marked drawn in queue.csv but not in the archive: "
                        f"{' '.join(sorted(queued - drawn))}")

    # 5. every output newer than its inputs
    arch, cj = mtime(RAW_FILES[0]), mtime("data/processed/counts.json")
    for cc in sorted(drawn):
        for kind in ("dots", "rings"):
            p = os.path.join(PROC, f"{kind}_{cc}.geojson")
            if os.path.exists(p) and os.path.getmtime(p) > cj:
                problems.append(f"{kind}_{cc}.geojson ({hhmm(os.path.getmtime(p))}) is newer than "
                                f"counts.json ({hhmm(cj)}) -- re-run the build tail")
    if cj > arch:
        problems.append(f"counts.json ({hhmm(cj)}) is newer than the archive ({hhmm(arch)}): "
                        f"tiles.py was interrupted, or --refresh-meta ran (fine if so)")
    if mtime("data/processed/not_drawn.geojson") < arch:
        problems.append("not_drawn.geojson is older than the archive: the tail stopped before "
                        "not_drawn.py")
    lj = mtime("taxonomy/languages.json")
    tax = os.path.join(ROOT, "taxonomy")
    for name in os.listdir(os.path.join(tax, "tree.d")):
        if name[:-4] in drawn and os.path.getmtime(os.path.join(tax, "tree.d", name)) > lj:
            problems.append(f"taxonomy/tree.d/{name} is newer than languages.json -- "
                            f"re-run the build tail")
    for name in os.listdir(tax):
        if name.endswith(".py") and name[:2] in drawn and name[2:3].isdigit() \
                and os.path.getmtime(os.path.join(tax, name)) > lj:
            problems.append(f"taxonomy/{name} is newer than languages.json -- re-run the build tail")

    # 6. every language node the archive draws has a label
    nodes = {n["id"]: n for n in json.load(open(os.path.join(tax, "languages.json"),
                                                encoding="utf-8"))["nodes"]}
    unlabelled = sorted({n for e in counts["countries"].values()
                         for part in ("dots", "rings") for n in e.get(part, {})
                         if not (nodes.get(n) or {}).get("label")})
    if unlabelled:
        problems.append(f"{len(unlabelled)} nodes in counts.json have no label in languages.json: "
                        f"{', '.join(unlabelled[:10])}{' ...' if len(unlabelled) > 10 else ''}")

    # 7. Auto's outlines cover every drawn country (tools/make_shapes.py; xs has no territory)
    shapes = json.load(open(os.path.join(ROOT, "country_shapes.geojson"), encoding="utf-8"))
    have = {f["properties"]["cc"] for f in shapes["features"]}
    if drawn - have - {"xs"}:
        problems.append(f"country_shapes.geojson has no outline for "
                        f"{' '.join(sorted(drawn - have - {'xs'}))} -- run tools/make_shapes.py")
    return problems, len(drawn)


def gzip_to(src, dst):
    """mtime=0 so the same bytes always give the same file, and --checksum skips it."""
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        with gzip.GzipFile(fileobj=f_out, mode="wb", compresslevel=9, mtime=0) as gz:
            shutil.copyfileobj(f_in, gz, 1 << 20)


def run(cmd, dry):
    print("   " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    return 0 if dry else subprocess.call(cmd)


def curl(args):
    r = subprocess.run(CURL + args, capture_output=True, text=True)
    return r.stdout.strip()


def verify():
    """The archive must answer a range request with 206 and must not be gzip-encoded; the
    counts must be gzip-encoded. Neither failure shows as an error on the page."""
    ok = True
    url = f"{PUBLIC}/{RAW_FILES[0]}"
    out = curl(["-r", "0-99", "-o", os.devnull, "-D", "-", "-w", "STATUS %{http_code}", url])
    code = out.rsplit("STATUS ", 1)[-1]
    enc = [l for l in out.splitlines() if l.lower().startswith("content-encoding")]
    print(f"   archive: {code}{', ' + enc[0] if enc else ''}")
    if code != "206":
        print("   NOT 206. Range requests are not being served: a basemap with no dots.")
        ok = False
    if enc:
        print("   THE ARCHIVE IS ENCODED and must not be. Re-upload it raw.")
        ok = False
    url = f"{PUBLIC}/data/processed/counts.json"
    out = curl(["-H", "Accept-Encoding: gzip", "-o", os.devnull, "-D", "-", url])
    enc = [l for l in out.splitlines() if l.lower().startswith("content-encoding")]
    print(f"   counts.json: {enc[0] if enc else 'no Content-Encoding'}")
    if not enc or "gzip" not in enc[0]:
        print("   NOT gzipped: the staging upload did not carry --header-upload.")
        ok = False
    return ok


def wrap_index(src, dst):
    """religiondots' web_wrap (Jekyll front matter + the site's analytics include), loaded by
    path and read-only; bytecode off so nothing is written into religiondots/."""
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "web_wrap", os.path.join(MAPS, "religiondots", "tools", "web_wrap.py"))
    ww = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ww)
    ww.wrap(src, dst, sitemap="false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-data", action="store_true",
                    help="only refresh index.html in the website repo")
    ap.add_argument("--verify", action="store_true", help="range-check the archive and exit")
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify() else 1)

    problems, n = preflight()
    if problems:
        print("PREFLIGHT FAILED" + (", nothing uploaded" if not a.dry_run else "") + ":")
        for p in problems:
            print("  - " + p)
        if not a.dry_run:
            sys.exit(1)
        print("\n(--dry-run: listing what a deploy would do anyway)\n")
    else:
        print(f"preflight ok: {n} countries, archive, counts, taxonomy and shapes agree\n")

    if not a.skip_data:
        if not shutil.which("rclone"):
            print("rclone is not on PATH. See COMMANDS.txt, DEPLOY.")
            if not a.dry_run:
                sys.exit(1)

        def fail(rc):
            print(f"   rclone exited {rc}. SignatureDoesNotMatch is a bad secret, `directory not "
                  f"found` a wrong bucket name, a 403 naming CreateBucket a missing "
                  f"--s3-no-check-bucket. religiondots/COMMANDS.txt, DEPLOY.")
            sys.exit(rc)

        print(f"raw -> {REMOTE}/")
        for rel in RAW_FILES:
            local = os.path.normpath(os.path.join(ROOT, *rel.split("/")))
            print(f"   ({os.path.getsize(local) / 1e6:.0f} MB)")
            rc = run(RCLONE + ["copyto", local, f"{REMOTE}/{rel}"], a.dry_run)
            if rc:
                fail(rc)

        print(f"gzipped via {os.path.relpath(STAGE, ROOT)}/ -> {REMOTE}/")
        for rel in GZ_FILES:
            src = os.path.join(ROOT, *rel.split("/"))
            dst = os.path.join(STAGE, *rel.split("/"))
            if not a.dry_run:
                gzip_to(src, dst)
            rc = run(RCLONE + GZ_HEADER + ["copyto", os.path.normpath(dst), f"{REMOTE}/{rel}"],
                     a.dry_run)
            if rc:
                fail(rc)
        print()

    print(f"repo files -> {WEBSITE}")
    for rel in REPO_FILES:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(WEBSITE, rel)
        print(f"   {rel}  (front matter + analytics via religiondots/tools/web_wrap.py"
              f"{', new folder' if not os.path.isdir(WEBSITE) else ''})")
        if not a.dry_run:
            wrap_index(src, dst)

    if a.dry_run:
        print(f"\nthen verify: curl -r 0-99 {PUBLIC}/{RAW_FILES[0]} must answer 206")
    elif not a.skip_data:
        print("\nverifying the published archive")
        verify()

    print("\nNOT COMMITTED. From the website repo:")
    print("   git add languagedots/")
    print('   git commit -m "languagedots"')
    print("   git push")
    print("\nThen: https://anita.garden/languagedots/")


if __name__ == "__main__":
    main()
