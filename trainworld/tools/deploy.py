"""Publish anitabuilder (T-924): the data to Cloudflare R2, the app to the website repo.

    python trainworld/tools/deploy.py --dry-run                     what would go where, how much
    python trainworld/tools/deploy.py --version 2026-10-22 --smoked  do it, then stop before git
    python trainworld/tools/deploy.py --verify --version 2026-10-22  check the published copy
    python trainworld/tools/deploy.py --stage DIR --data-base URL    the website half into DIR, for a local test

Run from the maps root. Nothing here commits or pushes; it prints the git commands.

What goes where. The build's `packs/` folder (city packs, market packs, outlines, the intercity
market, real saves, real-network graphs: about 1 GB in about 4,000 files) goes to
`r2:anitamaps/trainworld/<version>/`, read by the page from https://data.anita.garden/trainworld/<version>/.
Everything else in dist (index.html, assets/ with the JS, workers and WASM, fonts/, the favicon:
about 15 MB) goes to the website repo's `trainworld/`, since module workers and WASM must be
same-origin with the page. The page finds its data with game/dataBase.ts: the `tw-data` meta this
script writes into the published index.html, else `packs/` beside the page (every local build).

Versioned prefix. Every object is uploaded `immutable` for a year, so a later upload goes to a new
version and never mixes with a browser's cached old files; the script refuses a version already on
R2 unless --resume. Old versions can be deleted once the new page is live (COMMANDS in notes/T-924.md).

Compression. R2 compresses nothing. JSON and GeoJSON are uploaded gzipped with
`Content-Encoding: gzip` (179 MB of market packs alone). The .bin packs stay raw: the clock worker and
the walk grid read them with Range requests, and a range into a gzip stream is not a range into the
file (religiondots/tools/deploy.py). The .rgr graphs are gzip files the game inflates itself, so they
stay as they are with no Content-Encoding, or the browser would inflate them first.

`--s3-no-check-bucket` is required: without it rclone calls CreateBucket, which the token may not
do, and the copy fails with a 403 that looks like a dead token.
"""

import argparse
import datetime
import gzip
import json
import os
import shutil
import subprocess
import sys
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
TW = os.path.dirname(HERE)
MAPS = os.path.dirname(TW)
sys.path.insert(0, os.path.join(MAPS, "religiondots", "tools"))
import web_wrap  # noqa: E402  front matter and the site's analytics include

DIST = os.path.join(MAPS, ".tw-snap", "trainworld", "app", "dist")
WEBSITE = os.path.join(os.path.dirname(MAPS), "website")
STAGE = os.path.join(MAPS, "artifacts", "trainworld", "publish")
REMOTE = "r2:anitamaps/trainworld"
PUBLIC = "https://data.anita.garden/trainworld"
PAGE = "https://anita.garden/trainworld/"

RCLONE = ["rclone", "--s3-no-check-bucket", "--checksum", "--transfers", "8", "--stats-one-line", "--stats", "30s",
          "--header-upload", "Cache-Control: public, max-age=31536000, immutable"]
GZ = ["--header-upload", "Content-Encoding: gzip", "--header-upload", "Content-Type: application/json"]
GZ_EXT = (".json", ".geojson")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"
DTYPE = {"u8": 1, "i8": 1, "u16": 2, "i16": 2, "u32": 4, "i32": 4, "f32": 4, "u64": 8, "i64": 8, "f64": 8}
PACK_EXT = [".json", ".bin", ".boundary.geojson", ".places.json", ".market.json"]
MB = 1e6


def files_under(root):
    for dp, _, fn in os.walk(root):
        for f in fn:
            p = os.path.join(dp, f)
            yield os.path.relpath(p, root).replace("\\", "/"), p


def is_gz(rel):
    """JSON goes up gzipped, except the realgraph index the page also probes for locally (tiny)."""
    return rel.endswith(GZ_EXT) and not rel.startswith("realgraph/")


def git(*args, cwd):
    return subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True).stdout.strip()


def preflight(dist, allow_dirty, app=None):
    """The build's commit, the registry against the files, the sizes. Returns (problems, warnings, info)."""
    problems, warnings = [], []
    packs = os.path.join(dist, "packs")
    if not os.path.isfile(os.path.join(dist, "index.html")) or not os.path.isdir(packs):
        return [f"no build at {dist} (index.html and packs/)"], [], {}
    app = app or os.path.dirname(dist)
    repo = git("rev-parse", "--show-toplevel", cwd=app)
    head = git("rev-parse", "--short", "HEAD", cwd=app)
    dirty = git("status", "--porcelain", "--", "trainworld", cwd=app)
    if dirty and not allow_dirty:
        problems.append(f"{repo} has uncommitted trainworld changes; publish a build of a commit (the .tw-snap tree), or --allow-dirty")
    commit_t = int(git("log", "-1", "--format=%ct", cwd=app) or 0)
    built_t = os.path.getmtime(os.path.join(dist, "index.html"))
    if built_t < commit_t:
        problems.append(f"dist/index.html is older than {head}: rebuild before publishing")
    assets = os.path.join(dist, "assets")
    if not any("tw-data" in open(p, encoding="utf-8", errors="ignore").read()
               for p in (os.path.join(assets, f) for f in os.listdir(assets)) if p.endswith(".js")):
        problems.append("the build has no game/dataBase.ts (T-924): it would read packs/ beside the page and find nothing")

    reg = json.load(open(os.path.join(app, "src", "game", "registry.json"), encoding="utf-8"))
    ids = [c["id"] for c in reg["cities"]]
    missing = []
    for c in ids:
        for ext in PACK_EXT:
            if not os.path.isfile(os.path.join(packs, c + ext)):
                missing.append(c + ext)
        # every array the header names lies inside the .bin (a pack rewritten mid-copy shows here)
        try:
            h = json.load(open(os.path.join(packs, c + ".json"), encoding="utf-8"))
            size = os.path.getsize(os.path.join(packs, c + ".bin"))
            end = max((a["offset"] + a["count"] * DTYPE.get(a["dtype"], 8) for a in h.get("arrays", [])), default=0)
            if end > size:
                problems.append(f"{c}.json reaches byte {end} of a {size}-byte {c}.bin")
        except (OSError, ValueError, KeyError):
            pass
    for f in [reg["market"]["pack"] + ".json", "outlines.json", "realgraph/index.json"]:
        if not os.path.isfile(os.path.join(packs, f)):
            missing.append(f)
    try:
        for cc in json.load(open(os.path.join(packs, "realgraph", "index.json"), encoding="utf-8"))["regions"]:
            if not os.path.isfile(os.path.join(packs, "realgraph", cc + ".rgr")):
                missing.append(f"realgraph/{cc}.rgr")
    except (OSError, ValueError, KeyError):
        pass
    if missing:
        problems.append(f"{len(missing)} files the registry needs are not in the build: {', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")

    site = [(r, os.path.getsize(p)) for r, p in files_under(dist) if not r.startswith("packs/")]
    data = [(r, os.path.getsize(p)) for r, p in files_under(packs)]
    for r, s in site:
        if s > 100 * MB:
            problems.append(f"{r} is {s / MB:.0f} MB: over GitHub's 100 MB file limit")
        elif s > 50 * MB:
            warnings.append(f"{r} is {s / MB:.0f} MB: GitHub warns over 50 MB")
        if os.path.basename(r).startswith(("_", ".")):
            problems.append(f"{r}: Jekyll skips files starting with _ or .")
    site_total = sum(s for _, s in site)
    if site_total > 900 * MB:
        problems.append(f"the website part is {site_total / MB:.0f} MB, near Pages' 1 GB site limit")
    for r, s in data:
        if s > 100 * MB:
            warnings.append(f"packs/{r} is {s / MB:.0f} MB")
    return problems, warnings, {"commit": head, "repo": repo, "zones": len(ids), "site": site, "data": data}


def summary(info):
    print(f"build of {info['commit']} in {info['repo']}, {info['zones']} zones\n")
    groups = {}
    for r, s in info["data"]:
        k = r.split("/")[0] + "/" if "/" in r else next((e for e in [".boundary.geojson", ".places.json", ".market.json", ".bin"] if r.endswith(e)), r if r in ("outlines.json", "intercity.json") else ".json")
        g = groups.setdefault(k, [0, 0, 0])
        g[0] += 1
        g[1] += s
        g[2] += 0 if is_gz(r) else s
    print(f"R2 {REMOTE}/<version>/  ({PUBLIC}/<version>/)")
    for k, (n, s, raw) in sorted(groups.items(), key=lambda x: -x[1][1]):
        print(f"   {k:20s} {n:5d} files {s / MB:8.1f} MB  {'raw' if raw else 'gzipped'}")
    tot = sum(s for _, s in info["data"])
    print(f"   {'total':20s} {len(info['data']):5d} files {tot / MB:8.1f} MB\n")
    print(f"website {WEBSITE}{os.sep}trainworld{os.sep}")
    groups = {}
    for r, s in info["site"]:
        k = r.split("/")[0]
        g = groups.setdefault(k, [0, 0])
        g[0] += 1
        g[1] += s
    for k, (n, s) in sorted(groups.items(), key=lambda x: -x[1][1]):
        print(f"   {k:20s} {n:5d} files {s / MB:8.1f} MB")
    print(f"   {'total':20s} {len(info['site']):5d} files {sum(s for _, s in info['site']) / MB:8.1f} MB, largest {max(s for _, s in info['site']) / MB:.1f} MB\n")


def gzip_to(src, dst):
    """mtime 0, so the same bytes give the same file and --checksum skips it next time."""
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    with open(src, "rb") as fi, open(tmp, "wb") as fo, gzip.GzipFile(fileobj=fo, mode="wb", compresslevel=6, mtime=0) as gz:
        shutil.copyfileobj(fi, gz, 1 << 20)
    os.replace(tmp, dst)


def stage_gz(packs, version):
    out = os.path.join(STAGE, version, "gz")
    jobs = [(r, p) for r, p in files_under(packs) if is_gz(r)]
    raw = comp = 0
    for i, (r, p) in enumerate(jobs):
        d = os.path.join(out, *r.split("/"))
        gzip_to(p, d)
        raw += os.path.getsize(p)
        comp += os.path.getsize(d)
        if i % 500 == 0:
            print(f"   compressing {i}/{len(jobs)}", flush=True)
    print(f"   {len(jobs)} files, {raw / MB:.0f} MB -> {comp / MB:.0f} MB in {out}")
    return out


def run(cmd, dry):
    print("   " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if dry:
        return 0
    return subprocess.call(cmd)


def remote_versions():
    r = subprocess.run(["rclone", "lsf", REMOTE + "/", "--dirs-only", "--s3-no-check-bucket"], capture_output=True, text=True)
    if r.returncode:
        return None
    return [l.strip("/") for l in r.stdout.split()]


def page_html(dist, base):
    src = open(os.path.join(dist, "index.html"), encoding="utf-8").read()
    anchor = '<meta charset="utf-8" />'
    if anchor not in src:
        raise SystemExit("index.html has no <meta charset> to put the tw-data meta after")
    return src.replace(anchor, anchor + f'\n    <meta name="tw-data" content="{base}" />', 1)


def copy_site(dist, dest, base, dry):
    """dist minus packs/ into dest (the website's trainworld/), index.html wrapped for Jekyll."""
    print(f"   {dist} -> {dest}  (index.html with tw-data = {base})")
    if dry:
        return
    os.makedirs(dest, exist_ok=True)
    # old hashed chunks would pile up in the repo: clear the build's folders, keep index.html's front matter
    for d in ("assets", "fonts"):
        shutil.rmtree(os.path.join(dest, d), ignore_errors=True)
    for r, p in files_under(dist):
        if r.startswith("packs/") or r == "index.html":
            continue
        t = os.path.join(dest, *r.split("/"))
        os.makedirs(os.path.dirname(t), exist_ok=True)
        shutil.copy2(p, t)
    tmp = os.path.join(STAGE, "index.html")
    os.makedirs(STAGE, exist_ok=True)
    open(tmp, "w", encoding="utf-8", newline="").write(page_html(dist, base))
    web_wrap.wrap(tmp, os.path.join(dest, "index.html"), sitemap="false")


def fetch(url, headers):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **headers})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.status, r.headers, r.read(2000)


def verify(version, page):
    ok = True
    base = f"{PUBLIC}/{version}/"
    checks = [
        ("a .bin answers a Range request raw", base + "nyc.bin", {"Range": "bytes=0-99", "Origin": "https://anita.garden"},
         lambda c, h: c == 206 and not h.get("Content-Encoding") and h.get("Access-Control-Allow-Origin")),
        ("a pack header is gzipped and immutable", base + "nyc.json", {"Accept-Encoding": "gzip", "Origin": "https://anita.garden"},
         lambda c, h: h.get("Content-Encoding") == "gzip" and "immutable" in (h.get("Cache-Control") or "") and h.get("Access-Control-Allow-Origin")),
        ("the realgraph index is plain JSON", base + "realgraph/index.json", {"Origin": "https://anita.garden"},
         lambda c, h: c == 200 and not h.get("Content-Encoding")),
    ]
    if page:
        checks.append(("the page carries this version", PAGE, {}, None))
    for name, url, hd, test in checks:
        try:
            c, h, body = fetch(url, hd)
            good = (f"/{version}/" in body.decode("utf-8", "ignore")) if test is None else bool(test(c, h))
            print(f"   {'PASS' if good else 'FAIL'} {name}: {c}, encoding {h.get('Content-Encoding') or 'none'}, CORS {h.get('Access-Control-Allow-Origin') or 'none'}")
            ok &= good
        except Exception as e:  # noqa: BLE001
            print(f"   FAIL {name}: {url} {e}")
            ok = False
    if page:
        # module workers and WASM need the right types from Pages
        try:
            _, _, body = fetch(PAGE, {})
            html = body.decode("utf-8", "ignore")
            js = html.split('src="./assets/')[1].split('"')[0]
            c, h, _ = fetch(PAGE + "assets/" + js, {})
            print(f"   {'PASS' if 'javascript' in (h.get('Content-Type') or '') else 'FAIL'} entry script type {h.get('Content-Type')}")
        except Exception as e:  # noqa: BLE001
            print(f"   FAIL entry script: {e}")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--version", default=datetime.date.today().isoformat(), help="the R2 prefix, new for every upload (default today)")
    ap.add_argument("--dist", default=DIST, help="the build to publish (default the 8800 snapshot's)")
    ap.add_argument("--app", help="the app folder the build came from, for its commit and registry (default the dist's parent)")
    ap.add_argument("--smoked", action="store_true", help="smoke passed on this build (checks/smoke.mjs against 8800)")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--resume", action="store_true", help="upload into a version already on R2 (an interrupted upload)")
    ap.add_argument("--skip-data", action="store_true", help="only the website half (the data at --version is already up)")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--page", action="store_true", help="with --verify: also check the pushed page")
    ap.add_argument("--stage", help="write the website half into this folder instead (local test), nothing uploaded")
    ap.add_argument("--data-base", help="with --stage: the data URL the page reads")
    a = ap.parse_args()
    v = a.version

    if a.verify:
        sys.exit(0 if verify(v, a.page) else 1)

    problems, warnings, info = preflight(a.dist, a.allow_dirty or bool(a.stage), a.app)
    if info:
        summary(info)
    for w in warnings:
        print("warning: " + w)
    if problems:
        print("PREFLIGHT FAILED, nothing uploaded:")
        for p in problems:
            print("  - " + p)
        sys.exit(1)
    print("preflight ok\n")

    if a.stage:
        copy_site(a.dist, os.path.join(a.stage, "trainworld"), a.data_base or f"{PUBLIC}/{v}/", False)
        return

    if not a.dry_run and not a.smoked:
        print("Run smoke on this build first (node trainworld/checks/smoke.mjs http://localhost:8800/trainworld/), then pass --smoked.")
        sys.exit(1)
    if not shutil.which("rclone"):
        print("rclone is not on PATH")
        sys.exit(1)
    have = remote_versions()
    if have is None:
        print(f"could not list {REMOTE}/ (rclone remote r2: missing or no access)")
        if not a.dry_run:
            sys.exit(1)
    else:
        print(f"on R2 now: {', '.join(have) or 'no versions'}")
        if v in have and not (a.resume or a.skip_data):
            print(f"version {v} is already on R2: pick a new --version, or --resume an interrupted upload")
            if not a.dry_run:
                sys.exit(1)

    packs = os.path.join(a.dist, "packs")
    dest = f"{REMOTE}/{v}"
    if not a.skip_data:
        print(f"\ndata -> {dest}")
        # the raw half: .bin, .rgr, .save and the realgraph index (see is_gz)
        rc = run(RCLONE + ["copy", packs, dest, "--filter", "+ realgraph/index.json", "--filter", "- *.json", "--filter", "- *.geojson"], a.dry_run)
        if rc:
            sys.exit(f"rclone exited {rc}: a 403 naming CreateBucket is a missing --s3-no-check-bucket, SignatureDoesNotMatch a bad secret")
        if a.dry_run:
            n = sum(1 for r, _ in files_under(packs) if is_gz(r))
            print(f"   (would compress {n} JSON files into {os.path.join(STAGE, v, 'gz')})")
            gz = os.path.join(STAGE, v, "gz")
        else:
            gz = stage_gz(packs, v)
        rc = run(RCLONE + GZ + ["copy", gz, dest], a.dry_run)
        if rc:
            sys.exit(f"rclone exited {rc}")

    print("\nwebsite")
    copy_site(a.dist, os.path.join(WEBSITE, "trainworld"), f"{PUBLIC}/{v}/", a.dry_run)

    if not a.dry_run and not a.skip_data:
        print("\nverifying the data")
        verify(v, False)
    print("\nNOT COMMITTED. In the website repo:")
    print("   git -C ../website add trainworld/")
    print('   git -C ../website commit -m "anitabuilder"')
    print("   git -C ../website push")
    print(f"Then, once Pages has built: python trainworld/tools/deploy.py --verify --page --version {v}")
    print(f"and {PAGE}")


if __name__ == "__main__":
    main()
