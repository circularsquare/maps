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
import gzip
import os
import shutil
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import web_wrap

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEBSITE = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "website", "religiondots")

REMOTE = "r2:anitamaps/religiondots/data"
PUBLIC = "https://pub-ae551368cea941f39101e13c84d60bde.r2.dev/religiondots/data"

# What the viewer fetches out of data/.  `index.html`'s DATA_BASE builds every one of these
# paths, so a rename here is a rename there.
#
# *** R2 DOES NOT COMPRESS ANYTHING, AND NOTHING SAYS SO. ***  Ask it for counts.json with
# `Accept-Encoding: gzip` and it answers with all 665,494 bytes and no Content-Encoding at all.
# Every other map here is on GitHub Pages behind Cloudflare, which gzips text automatically --
# japanrail's 10.5 MB of segments crosses the wire at 2.5 MB -- so this is the one map paying
# the full price, and it was paying it on 66 MB.  The fix is to upload the bytes already
# compressed and say so in the object's metadata; browsers then inflate transparently and
# neither `.json()` nor `.arrayBuffer()` knows the difference.
#
# THE ARCHIVE IS THE ONE THING THAT MUST STAY RAW.  PMTiles reads it with byte-range requests,
# and a range into a gzip stream is not a range into the file -- this is the same failure as
# ancestrydots' original Pages diagnosis, arriving from the other direction.  Its tiles are
# individually compressed inside the archive anyway, so there is nothing to win.
RAW_FILES = ["processed/religiondots.pmtiles"]
GZ_FILES = ["processed/counts.json", "processed/country_shapes.geojson"]
GZ_DIRS = ["buffers"]

# Staged under data/, which is gitignored whole, so the compressed copies never reach a repo.
STAGE = "gz_stage"

# Copied into the website repo, keeping the same relative path.
REPO_FILES = ["index.html", "taxonomy/religions.json"]

# `--s3-no-check-bucket` IS NOT OPTIONAL AND ITS ABSENCE LOOKS LIKE A DEAD TOKEN.  Before its
# first upload rclone calls CreateBucket to make sure the destination exists, which is an ADMIN
# operation; the tokens here are Object Read & Write, so it is refused and the whole copy fails
# with `403 AccessDenied` naming CreateBucket.  Nothing is wrong with the token -- reads had
# already worked, because listing never calls it.  The flag says the bucket is known to exist.
# `--checksum` so an unchanged buffer is not re-uploaded: every build rewrites all 249 of them
# and only some of them actually differ.
RCLONE = ["rclone", "--s3-no-check-bucket", "--checksum", "--progress"]
GZ_HEADER = ["--header-upload", "Content-Encoding: gzip"]


def run(cmd, dry):
    print("   " + " ".join(cmd))
    if dry:
        return 0
    return subprocess.call(cmd)


def gzip_to(src, dst):
    """Compress src to dst, skipping the work when dst is already newer than src.

    MTIME ZERO, DELIBERATELY.  gzip writes the source's timestamp into its header, so
    compressing the same bytes twice gives two different files and two different checksums --
    and `--checksum` would then re-upload all 249 buffers on every deploy, which is the exact
    cost this staging exists to avoid.  Pinning it makes the output a pure function of the
    input.
    """
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as f_in, open(dst, "wb") as f_out:
        with gzip.GzipFile(fileobj=f_out, mode="wb", compresslevel=6, mtime=0) as gz:
            shutil.copyfileobj(f_in, gz, 1 << 20)
    return True


def stage_gzip(dry):
    """Mirror the compressible half of data/ into data/gz_stage/, under the SAME names.

    The names have to match, because the object key is what the viewer asks for; the file is
    gzip only in its bytes and its metadata, never in its path.
    """
    stage = os.path.join(ROOT, "data", STAGE)
    jobs = []
    for rel in GZ_FILES:
        jobs.append((os.path.join(ROOT, "data", *rel.split("/")),
                     os.path.join(stage, *rel.split("/"))))
    for d in GZ_DIRS:
        base = os.path.join(ROOT, "data", d)
        for name in sorted(os.listdir(base)):
            p = os.path.join(base, name)
            if os.path.isfile(p):
                jobs.append((p, os.path.join(stage, d, name)))
    if dry:
        print(f"   would compress {len(jobs)} files into data/{STAGE}/")
        return stage
    done = raw = comp = 0
    for src, dst in jobs:
        if gzip_to(src, dst):
            done += 1
        raw += os.path.getsize(src)
        comp += os.path.getsize(dst)
    print(f"   compressed {done} of {len(jobs)} files "
          f"({raw/1e6:.1f} MB -> {comp/1e6:.1f} MB, {comp/raw:.0%})")
    return stage


def preflight():
    """A half-finished build is the failure this catches, and it is invisible once uploaded.

    `build_tail.py` writes the archive, counts.json and the buffers in three steps under one
    lock.  If it was interrupted between them, every file exists and the set disagrees about
    which countries are drawn -- the viewer then greys out a country whose dots are sitting in
    the archive, or asks for a buffer that is not there and falls back to merged marks with no
    error.  Comparing the two country lists is the cheapest thing that notices.
    """
    problems = []
    for rel in RAW_FILES + GZ_FILES:
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
    """Two checks, and they pull in opposite directions.

    The archive must answer a RANGE request (206), which is what PMTiles reads it with, and it
    must NOT be gzipped, because a range into a gzip stream is not a range into the file.  The
    buffers must be the other way round: gzipped, or the map is downloading 59 MB where it
    could be downloading 39.  Neither failure shows on screen -- the first draws a basemap with
    no dots, the second just takes half a minute -- so both are asserted here rather than left
    to be noticed.
    """
    ok = True

    url = f"{PUBLIC}/processed/religiondots.pmtiles"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-99"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            code, n, enc = r.status, len(r.read()), r.headers.get("Content-Encoding")
        print(f"   archive: {code}, {n} bytes, encoding {enc or 'none'}")
        if code != 206 or n != 100:
            print("   NOT 206.  Range requests are not being served and the map will draw a "
                  "basemap with no dots.")
            ok = False
        if enc:
            print(f"   ARCHIVE IS {enc}-ENCODED and must not be.  Re-upload it out of "
                  f"RAW_FILES, not the staging tree.")
            ok = False
    except Exception as e:
        print(f"   FAILED {url}\n   {e}")
        ok = False

    # `Accept-Encoding: gzip` explicitly: urllib does not send one by default, and without it
    # a correctly gzipped object answers uncompressed and reads as a failure.
    url = f"{PUBLIC}/buffers/manifest.json"
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            enc = r.headers.get("Content-Encoding")
        print(f"   buffers: encoding {enc or 'none'}")
        if enc != "gzip":
            print("   NOT gzipped.  R2 compresses nothing on its own, so this means the "
                  "staging upload did not carry --header-upload.  Costs ~20 MB per load.")
            ok = False
    except Exception as e:
        print(f"   FAILED {url}\n   {e}")
        ok = False
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

        def fail(rc):
            print(f"   rclone exited {rc}.  Read WHICH error: SignatureDoesNotMatch is a "
                  f"bad secret, `directory not found` a wrong bucket name, a 403 naming "
                  f"CreateBucket a missing --s3-no-check-bucket.  COMMANDS.txt, DEPLOY.")
            sys.exit(rc)

        # The archive, uncompressed, on its own -- see RAW_FILES.
        for rel in RAW_FILES:
            sub = os.path.dirname(rel)
            local = os.path.normpath(os.path.join(ROOT, "data", *rel.split("/")))
            rc = run(RCLONE + ["copy", local, f"{REMOTE}/{sub}/"], a.dry_run)
            if rc:
                fail(rc)

        # Everything else, pre-compressed.  TWO transfers and never one against the remote
        # root: `processed/` also holds the archive, which the staging tree deliberately does
        # not contain, so a --prune sync from the root would DELETE it -- the whole 84 MB, for
        # a flag whose stated job is tidying away dead buffers.  Only `buffers/` is ever
        # synced, and it is the only prefix the staging tree is complete for.
        stage = stage_gzip(a.dry_run)
        rc = run(RCLONE + GZ_HEADER
                 + ["copy", os.path.normpath(os.path.join(stage, "processed")),
                    f"{REMOTE}/processed/"], a.dry_run)
        if rc:
            fail(rc)
        rc = run(RCLONE + GZ_HEADER
                 + ["sync" if a.prune else "copy",
                    os.path.normpath(os.path.join(stage, "buffers")),
                    f"{REMOTE}/buffers"], a.dry_run)
        if rc:
            fail(rc)
        print()

    print(f"repo files -> {WEBSITE}")
    for rel in REPO_FILES:
        src = os.path.join(ROOT, rel.replace("/", os.sep))
        dst = os.path.join(WEBSITE, rel.replace("/", os.sep))
        print(f"   {rel}")
        if a.dry_run:
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if rel == "index.html":
            # NOT a plain copy -- see web_wrap.  The deployed page needs Jekyll front matter
            # and the site's analytics include, and the working copy must not carry either,
            # because it has to run under a bare `npx serve` where Liquid does not exist.
            # Copying it flat drops the analytics with nothing to show for it.
            web_wrap.wrap(src, dst, sitemap="false")
        else:
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
