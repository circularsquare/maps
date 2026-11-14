#!/usr/bin/env python3
"""T-2913: build the public trainsworld repo from the maps tree, one way.

    python trainworld/tools/opensource/export.py [--out DIR] [--commit REF] [--no-commit] [--email ADDR]

Reads trainworld/ as committed at REF (default HEAD), so uncommitted edits and untracked files never travel.
Writes the folder (default C:/Users/anita/projects/trainsworld-export). A new folder is `git init`ed and gets one
commit. A folder from an earlier run keeps its .git (and its remote, once she adds one) and gets a "sync to" commit,
so every later sync is the same command. Touches nothing in maps, adds no remote, pushes nothing.

Layout: the public repo keeps the maps layout, `trainworld/` under the repo root. Paths in docs, checks and the
pipeline assume it (`node trainworld/checks/smoke.mjs`, `../../artifacts`), and an accepted pull request then
applies to maps with `git apply` unchanged.

What goes in is the allowlist INCLUDE minus EXCLUDE below; the repo-level files come from tools/opensource/files/.
Everything else (notes, TASKS, DECISIONS, ARCHIVE, TODO, research, saves of players, AWS tooling, the feedback
inbox, deploy, the sounds) stays out. A path under INCLUDE that no rule names is still exported: after changing
either list, read the printed tree summary.
"""
import argparse
import datetime
import fnmatch
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
MAPS = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
FILES = os.path.join(HERE, "files")

# paths under trainworld/ that are copied (prefix match on whole path parts)
INCLUDE = [
    "app", "sim", "pipeline", "checks", "docs", "wiki", "tools", "SPEC.md",
    "T-007-real-nyc.save", "T-125-northeast.save",  # the game's own test saves, no player data
]

# removed from INCLUDE; a trailing / means a folder, otherwise fnmatch on the whole path
EXCLUDE = [
    # sounds: Pixabay licence forbids handing the files out alone (research/release-legal.md); the game plays on without
    "app/public/sounds/",
    # the maps tree's own tooling, the friend's AWS account, her server and admin plumbing
    "tools/aws/", "tools/ops/", "tools/analytics/", "tools/feedback-inbox/", "tools/deploy.py", "tools/keep_assets.py",
    "tools/r2_purge.py", "tools/snap_wasm.py",
    "docs/aws.md", "docs/orchestrator.md", "docs/ops-checks.md",
    # one-offs and agent plumbing in checks/
    "checks/old/", "checks/fixtures/", "checks/t3302-player-saves.mjs",   # fixtures are saves a player sent us
    "checks/agent-tree.mjs", "checks/dev-nowatch.mjs", "checks/snapbuild*.cmd", "checks/smoke8800.cmd",
    "checks/mark_taking.py", "checks/keep_hunks.py", "checks/token_usage.py", "checks/runner_fixes.py",
    "checks/video-record.mjs",
    # generated or heavy and rebuilt locally
    "*/__pycache__/*", "*.pyc", "app/node_modules/", "app/dist/", "app/src/wasm/", "sim/target/",
    "wiki/.gitignore",
]

# files from tools/opensource/files/ copied to the repo: name -> path in the repo
EXTRA = {
    "README.md": "README.md",
    "LICENSE": "LICENSE",
    "NOTICE.md": "NOTICE.md",
    "CONTRIBUTING.md": "CONTRIBUTING.md",
    "gitignore": ".gitignore",
}


# checks that start her private server or use its admin plumbing are left out (the server is not in the public repo)
PRIVATE_MARKERS = (b"shonei.exe", b"shonei-server", b"SHONEI_", b"FEEDBACK_ADMIN")


def patch_vite(out):
    """The one code change the copy carries: a dev server that fetches the published data when there is no local
    data/packs (files/vite-data-fallback.ts). Fails loudly if vite.config.ts changed shape; then update the anchors."""
    path = os.path.join(out, "trainworld", "app", "vite.config.ts")
    src = open(path, encoding="utf-8").read()
    a1, a2 = "export default defineConfig({", "  plugins: [packs()],"
    if src.count(a1) != 1 or src.count(a2) != 1:
        sys.exit("vite.config.ts no longer has the anchors export.py patches: update patch_vite")
    frag = open(os.path.join(FILES, "vite-data-fallback.ts"), encoding="utf-8").read()
    src = src.replace(a1, frag + a1).replace(a2, "  plugins: [dataFallback(), packs()],")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(src)


def data_version(arg):
    """The published data folder: --data-version, else the one the website's trainsworld page names."""
    if arg:
        return arg
    page = os.path.abspath(os.path.join(MAPS, "..", "website", "trainsworld", "index.html"))
    try:
        m = re.search(r'name="tw-data" content="https://data\.anita\.garden/trainworld/([^/"]+)/"', open(page, encoding="utf-8").read())
        if m:
            return m.group(1)
    except OSError:
        pass
    sys.exit("no data version: the website page has no tw-data meta; pass --data-version")


def excluded(p):
    for r in EXCLUDE:
        if r.endswith("/"):
            if p.startswith(r):
                return True
        elif fnmatch.fnmatch(p, r):
            return True
    return False


def included(p):
    return any(p == i or p.startswith(i + "/") for i in INCLUDE) and not excluded(p)


def git(*a, cwd=MAPS, **kw):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\Users\anita\projects\trainsworld-export")
    ap.add_argument("--commit", default="HEAD", help="the maps commit to export")
    ap.add_argument("--data-version", default=None, help="published data folder on R2 (default: the website page's)")
    ap.add_argument("--no-commit", action="store_true", help="write the tree, skip git init and the commit")
    ap.add_argument("--email", default=None, help="author email of the first commit (default: maps' git config)")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    if out.startswith(MAPS + os.sep) or out == MAPS:
        sys.exit("refusing: the export folder must be outside the maps repo")

    sha = git("rev-parse", "--short=8", a.commit).stdout.decode().strip()
    full = git("rev-parse", a.commit).stdout.decode().strip()
    sync = os.path.isdir(os.path.join(out, ".git")) and os.path.exists(os.path.join(out, "trainworld", "SPEC.md"))
    if os.path.exists(out):
        # a folder from an earlier run keeps its .git and the files are rewritten, so the next commit is a sync;
        # any other non-empty folder is refused
        if not sync and os.listdir(out):
            sys.exit(f"refusing: {out} exists and is not a previous export")
        def rm(fn, path, exc):
            os.chmod(path, 0o700); fn(path)
        for name in os.listdir(out):
            if name != ".git":
                path = os.path.join(out, name)
                shutil.rmtree(path, onerror=rm) if os.path.isdir(path) else os.remove(path)
    os.makedirs(out, exist_ok=True)

    tar_bytes = git("archive", "--format=tar", a.commit, "trainworld", cwd=MAPS).stdout
    n = size = 0
    kept = {}
    skipped = []
    with tarfile.open(fileobj=io.BytesIO(tar_bytes)) as tf:
        for m in tf:
            if not m.isfile():
                continue
            rel = m.name[len("trainworld/"):]
            if not included(rel):
                continue
            data = tf.extractfile(m).read()
            if rel.startswith("checks/") and any(k in data for k in PRIVATE_MARKERS):
                skipped.append(rel)
                continue
            dst = os.path.join(out, "trainworld", *rel.split("/"))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as f:
                f.write(data)
            n += 1
            size += m.size
            top = rel.split("/")[0]
            kept[top] = kept.get(top, 0) + m.size
    patch_vite(out)
    ver = data_version(a.data_version)
    with open(os.path.join(out, "trainworld", "data-version.txt"), "w", newline="\n") as f:
        f.write(ver + "\n")
    print(f"data version {ver}")
    if skipped:
        print(f"left out {len(skipped)} checks that need her private server: " + ", ".join(os.path.basename(x) for x in skipped))
    for src, dst in EXTRA.items():
        s = os.path.join(FILES, src)
        if not os.path.exists(s):
            print(f"  missing extra file {src}")
            continue
        d = os.path.join(out, *dst.split("/"))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copyfile(s, d)
        n += 1
        size += os.path.getsize(d)
    print(f"exported {n} files, {size / 1e6:.1f} MB from maps {sha} to {out}")
    for k, v in sorted(kept.items(), key=lambda kv: -kv[1]):
        print(f"  {k:28} {v / 1e6:7.1f} MB")

    if a.no_commit:
        return
    env = dict(os.environ)
    name = git("config", "user.name").stdout.decode().strip()
    email = a.email or git("config", "user.email").stdout.decode().strip()
    env.update(GIT_AUTHOR_NAME=name, GIT_COMMITTER_NAME=name, GIT_AUTHOR_EMAIL=email, GIT_COMMITTER_EMAIL=email)
    run = lambda *c: subprocess.run(["git", *c], cwd=out, check=True, env=env, capture_output=True)
    if not sync:
        run("init", "-q", "-b", "main")
        run("config", "core.autocrlf", "false")
    run("add", "-A")
    if sync:
        if not subprocess.run(["git", "status", "--porcelain"], cwd=out, capture_output=True, text=True).stdout.strip():
            print("nothing changed since the last sync")
            return
        msg = f"sync to maps {sha} ({datetime.date.today().isoformat()}).\n\nSource commit {full}."
        # contributors credited in maps since the last sync are credited on the sync commit too (CONTRIBUTING.md)
        last = subprocess.run(["git", "log", "-1", "--format=%b"], cwd=out, capture_output=True, text=True).stdout
        prev = re.search(r"Source commit ([0-9a-f]{40})", last)
        if prev:
            bodies = git("log", "--format=%b", f"{prev.group(1)}..{full}").stdout.decode()
            credits = sorted({l.strip() for l in bodies.splitlines()
                              if l.strip().lower().startswith("co-authored-by:") and "anthropic.com" not in l})
            if credits:
                msg += "\n\n" + "\n".join(credits)
    else:
        msg = (f"trainsworld: first public copy, from the private maps tree at {sha} ({datetime.date.today().isoformat()}).\n\n"
               f"Source commit {full}. History stays private.")
    run("commit", "-q", "-m", msg)
    print("committed in the export folder:", subprocess.run(["git", "log", "--oneline", "-1"], cwd=out, capture_output=True, text=True).stdout.strip())


if __name__ == "__main__":
    main()
