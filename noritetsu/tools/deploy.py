"""Publish noritetsu to anita.garden/noritetsu: the data to Cloudflare R2, the page to the website repo.

    python tools/deploy.py --check       the preflight only: what is built, what is missing
    python tools/deploy.py --dry-run     preflight, then what rclone would send and delete
    python tools/deploy.py               preflight, upload, copy the page files, verify; no git
    python tools/deploy.py --verify      check the published copy on its own
    python tools/deploy.py --skip-data   only copy the page files into the website repo

WHAT GOES WHERE. Everything the page fetches goes to R2, under r2:anitamaps/noritetsu/:
`regions.json` and all of `dist/data/` (the .pmtiles, every country's json and geom/, logos/).
The website repo gets the page and its two scripts: `index.html`, `poster.js`, `share.js`.
The page decides where its data is by asking for `regions.json` beside itself (index.html,
"WHERE THE DATA IS"), so **regions.json must never be copied into the website repo**: if it
were, the published page would look for data/ on GitHub Pages and find none. The preflight
refuses to run while the website copy has a regions.json or a data/.

R2 RATHER THAN PAGES IS NOT ABOUT SIZE. PMTiles reads with HTTP range requests, and GitHub
Pages fronted by Cloudflare breaks them: the map then loads a basemap and no rail at all.
`--verify` asks for `bytes=0-99` and it must answer 206.

R2 COMPRESSES NOTHING, so the json goes up gzipped with `Content-Encoding: gzip` (religiondots'
tools/deploy.py found this; the browser inflates it and `.json()` never knows). It is staged
in `data/deploy_gz/` (gitignored, under the same names) with gzip's mtime pinned to zero, so the
same input always gives the same bytes and `--checksum` only sends what changed. The .pmtiles
and .png go up raw: a range into a gzip stream is not a range into the file.

TWO SYNCS, EACH FILTERED, SO NEITHER CAN DELETE THE OTHER'S FILES. The json sync only sees
`*.json` on both sides, the raw sync everything else, and both stay under noritetsu/data, so a
file the build no longer writes (a line id that went away) is deleted from R2 and nothing
outside the prefix is touched. Everything is uploaded with `Cache-Control: no-cache`: the
browser keeps its copy but asks each time whether it changed, so a redeploy is seen at once
and a page never mixes one build's lines with another's geometry.

THIS SCRIPT NEVER COMMITS OR PUSHES; it prints the git commands and stops.
"""

import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zlib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")
DATA = os.path.join(DIST, "data")
STAGE = os.path.join(ROOT, "data", "deploy_gz")
LOGS = os.path.join(ROOT, "data", "logs")
WEBSITE = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "website", "noritetsu")

REMOTE = "r2:anitamaps/noritetsu"
PUBLIC = "https://pub-ae551368cea941f39101e13c84d60bde.r2.dev/noritetsu"

# Every country in regions.json needs these beside its tiles. foot.json above all: without it
# the app credits that country's OSM lines whole (HANDOFF, "Publishing").
REGION_FILES = ["lines.json", "stations.json", "foot.json", "ways.json", "aliases.json",
                "types.json"]
# One file for every country; the app degrades without them, and quietly.
SHARED_FILES = ["search.json", "closed.json", "operators.json"]
# Wanted but not fatal: the app has a fallback for each.
SHARED_OPTIONAL = ["names_en.json", "riders_sources.json"]
PAGE_FILES = ["index.html", "poster.js", "share.js"]

# `--s3-no-check-bucket`: without it rclone asks to CreateBucket first, an admin call the
# Object Read & Write token is refused, and the copy dies with a 403 that reads like a dead
# token (reference_map_publishing). `--fast-list` lists the 22,000 objects in a few calls.
RCLONE = ["rclone", "--s3-no-check-bucket", "--checksum", "--fast-list",
          "--transfers", "16", "--checkers", "16", "--stats", "30s", "--stats-one-line",
          "--header-upload", "Cache-Control: no-cache"]
GZ_HEADER = ["--header-upload", "Content-Encoding: gzip"]

# r2.dev is behind Cloudflare's bot rules and 403s `Python-urllib`, which looks exactly like a
# failed upload (religiondots' deploy.py). A plain browser string, nothing that names anyone.
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0 Safari/537.36")

# A build writing into dist/ while it uploads would publish half of it.
BUILD_SCRIPTS = re.compile(r"build_model|build_tiles|build_regions|rebuild\.py|along\.py|"
                           r"operators\.py|english_names|line_types|station_riders|extract\.py")


def mb(n):
    return f"{n / 1e6:,.1f} MB"


def busy_reasons(quiet_min):
    """Signs that a build is running or has just written: the preflight stops on any of them."""
    out = []
    now = time.time()
    # 1. Build output (dist/data, dist/regions.json) written in the last few minutes. The page
    #    files are left out: an edit to index.html is not a build.
    newest, newest_p = os.path.getmtime(os.path.join(DIST, "regions.json")), \
        os.path.join(DIST, "regions.json")
    for root, _, files in os.walk(DATA):
        for f in files:
            m = os.path.getmtime(os.path.join(root, f))
            if m > newest:
                newest, newest_p = m, os.path.join(root, f)
    if newest and now - newest < quiet_min * 60:
        out.append(f"{os.path.relpath(newest_p, ROOT)} was written {(now - newest) / 60:.1f} min "
                   f"ago (quiet window {quiet_min} min)")
    # 2. Build logs still being written (tools/rebuild.py, data/logs/rebuild_<cc>_<step>.txt).
    if os.path.isdir(LOGS):
        for f in os.listdir(LOGS):
            p = os.path.join(LOGS, f)
            if f.endswith(".txt") and os.path.isfile(p) and now - os.path.getmtime(p) < quiet_min * 60:
                out.append(f"data/logs/{f} was written {(now - os.path.getmtime(p)) / 60:.1f} min ago")
    # 3. A CPU slot held (tools/slot.py): somebody's build, extract or trial is running.
    slots = os.path.join(LOGS, "slots")
    if os.path.isdir(slots) and sys.platform == "win32":
        import msvcrt
        for f in sorted(os.listdir(slots)):
            if not f.endswith(".lock"):
                continue
            # "a+b", never "w+b": opening another holder's lock must not truncate it.
            with open(os.path.join(slots, f), "a+b") as h:
                h.seek(0)
                try:
                    msvcrt.locking(h.fileno(), msvcrt.LK_NBLCK, 1)
                    h.seek(0)
                    msvcrt.locking(h.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    out.append(f"CPU slot {f} is held (tools/slot.py): a build is running")
    # 4. A build script running right now, whoever started it.
    if sys.platform == "win32":
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-Command",
                                "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
                                "| ForEach-Object { \"$($_.ProcessId) $($_.CommandLine)\" }"],
                               capture_output=True, text=True, timeout=60)
            for line in r.stdout.splitlines():
                if BUILD_SCRIPTS.search(line) and "deploy.py" not in line:
                    out.append(f"running: {line.strip()[:160]}")
        except Exception as e:
            print(f"   (could not list processes: {e})")
    return out


def preflight(quiet_min, ignore_busy):
    problems, warnings = [], []

    try:
        reg = json.load(open(os.path.join(DIST, "regions.json"), encoding="utf-8"))
        regions = sorted(reg["regions"])
    except Exception as e:
        return [f"dist/regions.json does not load: {e} -- run tools/build_regions.py"], [], None

    for cc in regions:
        d = os.path.join(DATA, cc)
        for f in REGION_FILES:
            p = os.path.join(d, f)
            if not os.path.isfile(p) or os.path.getsize(p) == 0:
                problems.append(f"{cc}: data/{cc}/{f} missing or empty")
        pm = os.path.join(DATA, f"{cc}.pmtiles")
        if not os.path.isfile(pm) or os.path.getsize(pm) < 1000:
            problems.append(f"{cc}: data/{cc}.pmtiles missing or empty -- build_tiles.py")
        g = os.path.join(d, "geom")
        if not os.path.isdir(g) or not os.listdir(g):
            problems.append(f"{cc}: data/{cc}/geom/ missing or empty")
        # The old crediting file. foot.json replaced it, and a stale one must not ship.
        if os.path.exists(os.path.join(d, "credits.json")):
            problems.append(f"{cc}: data/{cc}/credits.json is the old crediting file; delete it "
                            f"(foot.json replaced it)")

    built = sorted(x for x in os.listdir(DATA)
                   if os.path.isdir(os.path.join(DATA, x)) and x != "logos")
    extra = sorted(set(built) - set(regions))
    if extra:
        warnings.append(f"built but not in regions.json (uploaded, never loaded): "
                        f"{', '.join(extra)} -- run tools/build_regions.py?")

    for f in SHARED_FILES:
        if not os.path.isfile(os.path.join(DATA, f)):
            problems.append(f"data/{f} missing -- " + ("tools/operators.py build" if
                            f == "operators.json" else "tools/build_regions.py"))
    for f in SHARED_OPTIONAL:
        if not os.path.isfile(os.path.join(DATA, f)):
            warnings.append(f"data/{f} missing (the app works without it)")

    # Every logo operators.json names is on disk. (logos/credits.json is the logos' own
    # attribution and does ship; it is not the old crediting file.)
    try:
        ops = json.load(open(os.path.join(DATA, "operators.json"), encoding="utf-8"))
        for k, v in ops.items():
            logo = isinstance(v, dict) and v.get("logo")
            if logo and not os.path.isfile(os.path.join(DIST, *logo.split("/"))):
                problems.append(f"operators.json names {logo} for {k}, not on disk")
    except Exception as e:
        if os.path.isfile(os.path.join(DATA, "operators.json")):
            problems.append(f"data/operators.json does not load: {e}")

    for f in PAGE_FILES:
        if not os.path.isfile(os.path.join(DIST, f)):
            problems.append(f"dist/{f} missing")
    page = open(os.path.join(DIST, "index.html"), encoding="utf-8").read()
    if "DATA_R2" not in page or "dataFetch" not in page:
        problems.append("index.html has no data-base switch (DATA_R2 / dataFetch): the "
                        "published page would look for data/ on GitHub Pages")
    # Jekyll runs the page through Liquid; `{{` or `{%` in it would be eaten.
    for tok in ("{{", "{%"):
        if tok in page:
            n = page[:page.index(tok)].count("\n") + 1
            problems.append(f"index.html line {n} has `{tok}`, which Jekyll's Liquid would eat")
    lint = subprocess.run(["node", os.path.join(ROOT, "tools", "lint_map_expressions.js"),
                           os.path.join(DIST, "index.html")], capture_output=True, text=True)
    if lint.returncode:
        problems.append("tools/lint_map_expressions.js fails:\n      "
                        + (lint.stdout + lint.stderr).strip().replace("\n", "\n      "))
    else:
        print(f"   {lint.stdout.strip()}")

    for bad in ("regions.json", "data"):
        if os.path.exists(os.path.join(WEBSITE, bad)):
            problems.append(f"website/noritetsu/{bad} exists: the page would read its data from "
                            f"GitHub Pages instead of R2. Remove it.")

    busy = busy_reasons(quiet_min)
    if busy and ignore_busy:
        warnings += [f"busy, ignored (--ignore-busy): {b}" for b in busy]
    else:
        problems += [f"busy: {b}" for b in busy]
    return problems, warnings, regions


def size_summary():
    groups = {}
    biggest = []
    for root, _, files in os.walk(DATA):
        for f in files:
            p = os.path.join(root, f)
            n = os.path.getsize(p)
            rel = os.path.relpath(p, DATA).replace(os.sep, "/")
            key = ("tiles (.pmtiles)" if f.endswith(".pmtiles") else
                   "logos/" if rel.startswith("logos/") else
                   "geom/<line>.json" if "/geom/" in rel else
                   "country json" if "/" in rel else "shared json")
            c, b = groups.get(key, (0, 0))
            groups[key] = (c + 1, b + n)
            biggest.append((n, rel))
    tot_c = sum(c for c, _ in groups.values())
    tot_b = sum(b for _, b in groups.values())
    print(f"   dist/data: {tot_c:,} files, {mb(tot_b)}")
    for k, (c, b) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        print(f"     {k:18s} {c:7,} files  {mb(b):>10s}")
    biggest.sort(reverse=True)
    print("   biggest: " + ", ".join(f"{r} {mb(n)}" for n, r in biggest[:5]))
    return tot_c, tot_b


def gzip_to(src, dst):
    """Compress src to dst unless dst is already newer. mtime=0 so the bytes are a pure
    function of the input and --checksum does not resend an unchanged file."""
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        with gzip.GzipFile(filename="", fileobj=fo, mode="wb", compresslevel=6, mtime=0) as gz:
            shutil.copyfileobj(fi, gz, 1 << 20)
    os.replace(tmp, dst)
    return True


def stage_gzip():
    """Mirror every .json the page fetches into data/deploy_gz/ under the same names, and drop
    staged files whose source is gone, so the json sync deletes them from R2 too."""
    jobs = [(os.path.join(DIST, "regions.json"), os.path.join(STAGE, "regions.json"))]
    for root, _, files in os.walk(DATA):
        for f in files:
            if f.endswith(".json"):
                src = os.path.join(root, f)
                jobs.append((src, os.path.join(STAGE, "data", os.path.relpath(src, DATA))))
    want = {os.path.normcase(d) for _, d in jobs}
    gone = 0
    if os.path.isdir(STAGE):
        for root, _, files in os.walk(STAGE):
            for f in files:
                p = os.path.join(root, f)
                if os.path.normcase(p) not in want:
                    os.remove(p)
                    gone += 1
    t0 = time.time()
    done = raw = comp = 0
    for i, (src, dst) in enumerate(jobs):
        if gzip_to(src, dst):
            done += 1
        raw += os.path.getsize(src)
        comp += os.path.getsize(dst)
        if (i + 1) % 5000 == 0:
            print(f"   ... {i + 1:,} of {len(jobs):,}", flush=True)
    what = f"compressed {done:,} of {len(jobs):,}" if done else f"all {len(jobs):,} already current"
    print(f"   {what} ({mb(raw)} -> {mb(comp)}, {comp / raw:.0%}), {gone} stale removed, "
          f"{time.time() - t0:.0f} s")


def run(cmd):
    print("   " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    return subprocess.call(cmd)


def upload(dry):
    flag = ["--dry-run"] if dry else []

    def fail(rc):
        print(f"   rclone exited {rc}. Read WHICH error: SignatureDoesNotMatch is a bad secret, "
              f"`directory not found` a wrong bucket name, a 403 naming CreateBucket a missing "
              f"--s3-no-check-bucket, any other 403 AccessDenied an expired or revoked token. "
              f"README.md, Deploy.")
        sys.exit(rc)

    print(f"\nraw (.pmtiles, logos) -> {REMOTE}/data")
    rc = run(RCLONE + flag + ["sync", DATA, f"{REMOTE}/data", "--exclude", "*.json"])
    if rc:
        fail(rc)
    print(f"\njson, gzipped -> {REMOTE}/data")
    rc = run(RCLONE + GZ_HEADER + flag + ["sync", os.path.join(STAGE, "data"), f"{REMOTE}/data",
                                          "--include", "*.json"])
    if rc:
        fail(rc)
    # Last, so a page loading mid-deploy never has an index naming a country not up yet.
    print(f"\nregions.json, gzipped -> {REMOTE}/regions.json")
    rc = run(RCLONE + GZ_HEADER + flag + ["copyto", os.path.join(STAGE, "regions.json"),
                                          f"{REMOTE}/regions.json"])
    if rc:
        fail(rc)


def get(url, headers):
    h = {"User-Agent": UA}
    h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, dict(r.headers), r.read()


def verify():
    """The archive must answer a range request (206) and not be gzipped; the json must be
    gzipped and inflate to the local file's size; CORS must answer anita.garden. None of these
    failures shows on screen as what it is."""
    ok = True
    origin = {"Origin": "https://anita.garden"}

    for cc in ("de", "jp"):
        url = f"{PUBLIC}/data/{cc}.pmtiles"
        try:
            code, hd, body = get(url, {"Range": "bytes=0-99", **origin})
            enc, acao = hd.get("Content-Encoding"), hd.get("Access-Control-Allow-Origin")
            print(f"   {cc}.pmtiles range: {code}, {len(body)} bytes, encoding {enc or 'none'}, "
                  f"CORS {acao or 'NONE'}")
            if code != 206 or len(body) != 100 or enc or not acao:
                print("   WRONG: needs 206, 100 bytes, no encoding, and a CORS header.")
                ok = False
            if body[:7] != b"PMTiles":
                print(f"   WRONG: does not start with the PMTiles magic: {body[:7]!r}")
                ok = False
        except Exception as e:
            print(f"   FAILED {url}: {e}")
            ok = False

    for rel in ("regions.json", "data/search.json", "data/closed.json", "data/operators.json",
                "data/de/lines.json", "data/jp/foot.json"):
        url = f"{PUBLIC}/{rel}"
        local = os.path.join(DIST, *rel.split("/"))
        try:
            code, hd, body = get(url, {"Accept-Encoding": "gzip", **origin})
            enc, acao = hd.get("Content-Encoding"), hd.get("Access-Control-Allow-Origin")
            raw = zlib.decompress(body, 16 + zlib.MAX_WBITS) if enc == "gzip" else body
            same = len(raw) == os.path.getsize(local)
            print(f"   {rel}: {code}, {mb(len(body))} on the wire, {mb(len(raw))} inflated "
                  f"({'= local' if same else 'LOCAL IS ' + mb(os.path.getsize(local))}), "
                  f"encoding {enc or 'none'}, CORS {acao or 'NONE'}")
            if code != 200 or enc != "gzip" or not acao:
                ok = False
            if not same:
                print("   differs from dist/: a rebuild since the upload, or the upload is stale")
                ok = False
        except Exception as e:
            print(f"   FAILED {url}: {e}")
            ok = False

    ops = json.load(open(os.path.join(DATA, "operators.json"), encoding="utf-8"))
    logo = next((v["logo"] for v in ops.values() if isinstance(v, dict) and v.get("logo")), None)
    if logo:
        try:
            code, hd, body = get(f"{PUBLIC}/{logo}", origin)
            print(f"   {logo}: {code}, {len(body)} bytes, {hd.get('Content-Type')}")
            ok = ok and code == 200
        except Exception as e:
            print(f"   FAILED {logo}: {e}")
            ok = False
    print("   verify " + ("OK" if ok else "FAILED"))
    return ok


def wrap_page(src, dst):
    """The page with the site's Jekyll front matter and analytics include, as every page in the
    website repo has them (religiondots/tools/web_wrap.py). Front matter already at the
    destination is kept, since it carries per-page settings; new pages stay out of the sitemap."""
    text = open(src, encoding="utf-8", newline="").read()     # line endings as they are
    fm = "---\nsitemap: false\n---\n"
    if os.path.exists(dst):
        head = open(dst, encoding="utf-8").read(4096)
        end = head.find("\n---", 3)
        if head.startswith("---") and end > 0:
            fm = head[:end + 4].rstrip("\n") + "\n"
    inc = "{%- include analytics.html -%}"
    nl = "\r\n" if "\r\n" in text else "\n"
    m = re.search(r'^([ \t]*)<meta\s+charset=[^>]*>[ \t]*(?=\r?$)', text, re.I | re.M)
    if not m:
        sys.exit(f"{src}: no <meta charset> to put the analytics include after")
    out = fm.replace("\n", nl) + text[:m.end()] + nl + m.group(1) + inc + text[m.end():]
    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.write(out)


def copy_page(dry):
    print(f"\npage -> {WEBSITE}")
    for f in PAGE_FILES:
        print(f"   {f}")
        if dry:
            continue
        os.makedirs(WEBSITE, exist_ok=True)
        if f == "index.html":
            wrap_page(os.path.join(DIST, f), os.path.join(WEBSITE, f))
        else:
            shutil.copy2(os.path.join(DIST, f), os.path.join(WEBSITE, f))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="preflight and sizes only")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--skip-data", action="store_true", help="only the page files")
    ap.add_argument("--quiet-minutes", type=float, default=10,
                    help="dist/ and the build logs must be this long untouched (default 10)")
    ap.add_argument("--ignore-busy", action="store_true",
                    help="upload although a build looks active: only once you have checked "
                         "that what is running does not write dist/")
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify() else 1)

    print("preflight")
    problems, warnings, regions = preflight(a.quiet_minutes, a.ignore_busy)
    size_summary()
    for w in warnings:
        print(f"   warning: {w}")
    if problems:
        print(f"\nPREFLIGHT FAILED ({len(problems)}), nothing uploaded:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print(f"   preflight ok: {len(regions)} countries")
    if a.check:
        return

    if not a.skip_data:
        if not shutil.which("rclone"):
            sys.exit("rclone is not on PATH")
        print(f"\nstaging gzipped json in {os.path.relpath(STAGE, ROOT)}")
        stage_gzip()
        t0 = time.time()
        upload(a.dry_run)
        print(f"\n   rclone {'dry run' if a.dry_run else 'upload'}: {(time.time() - t0) / 60:.1f} min")

    copy_page(a.dry_run)

    if not a.dry_run and not a.skip_data:
        print("\nverifying the published copy")
        verify()

    print("\nNOT COMMITTED. In the website repo:")
    print("   git add noritetsu/")
    print('   git commit -m "noritetsu"')
    print("   git push")
    print("Then: https://anita.garden/noritetsu/")


if __name__ == "__main__":
    main()
