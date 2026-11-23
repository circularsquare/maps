r"""Rebuild 8800's track build (T-4126): tw/track's tip, built with TW_CHANNEL=track in its own checkout
(maps/.tw-trees/tracksnap, made on first use) and served at http://localhost:8800/trainworld-track/, which
http://localhost:8800/trainworld/?track opens. Its saves, settings and sign-in are its own (app/src/game/trackBuild.ts),
so nothing it saves reaches the main build's saves list or the account. 8800's main build (.tw-snap) is never touched.

Usage, from the maps root (about 3 minutes, longer when sim/ changed since the last run):
    venv\Scripts\python.exe trainworld/tools/ops/snaptrack.py [<commit>] [--as-is] [--keep-wasm] [--log <file>]

<commit> defaults to tw/track's tip. --as-is builds the checkout as it stands (no checkout first); --keep-wasm uses the
WASM already in it. A commit without the track build's own storage is refused: merge master into tw/track first.
"""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# the main tree, also when this copy of the script sits in a worktree
_HERE = Path(__file__).resolve().parent
MAPS = (_HERE / subprocess.run(["git", "-C", str(_HERE), "rev-parse", "--git-common-dir"],
                               capture_output=True, text=True, creationflags=0x08000000).stdout.strip()).resolve().parent
TREE = MAPS / ".tw-trees" / "tracksnap"
APP = TREE / "trainworld" / "app"
NAME = "trainworld-track"
NO_WINDOW = 0x08000000


def run(cmd, cwd, env=None, check=True):
    cmd = [str(c) for c in cmd]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd), env=env, creationflags=NO_WINDOW, shell=(os.name == "nt" and cmd[0] in ("npx", "node")))
    if check and r.returncode:
        sys.exit(f"failed (exit {r.returncode}): {' '.join(cmd)}")
    return r.returncode


def git(*args, cwd=MAPS):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, creationflags=NO_WINDOW).stdout.strip()


def main(argv):
    log = None
    if "--log" in argv:
        i = argv.index("--log")
        log = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if log:
        f = open(log, "w", encoding="utf-8", errors="replace")
        sys.stdout = sys.stderr = f
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
    as_is, keep_wasm = "--as-is" in argv, "--keep-wasm" in argv
    rest = [a for a in argv if not a.startswith("--")]
    commit = git("rev-parse", rest[0] if rest else "tw/track")
    if not commit:
        sys.exit(f"no such commit: {rest[0] if rest else 'tw/track'}")
    if not TREE.exists():
        run(["node", MAPS / "trainworld" / "checks" / "agent-tree.mjs", "new", "tracksnap", commit], MAPS)
    if not as_is:
        run(["git", "-C", TREE, "checkout", "--detach", "--force", commit], MAPS)
    built = git("rev-parse", "HEAD", cwd=TREE)
    # without its own storage a track build would read and write the main build's saves on the same origin
    if not (APP / "src" / "game" / "trackBuild.ts").exists() or "TW_CHANNEL" not in (APP / "vite.config.ts").read_text(encoding="utf-8"):
        sys.exit(f"{built[:8]} has no track build storage (T-4126): merge master into tw/track first (agent-tree.mjs sync-track)")
    env = dict(os.environ, TW_CHANNEL="track", TW_LINK_PACKS="1", CARGO_BUILD_JOBS="2")
    env["PATH"] = str(Path.home() / ".cargo" / "bin") + os.pathsep + env["PATH"]
    if not keep_wasm:
        claim = subprocess.run([sys.executable, str(MAPS / "cores.py"), "claim", "4", "snaptrack", "--pid", str(os.getpid()), "--min", "2", "track build WASM"],
                               cwd=str(MAPS), creationflags=NO_WINDOW)
        if claim.returncode:
            sys.exit("no cores from the ledger; not building")
        try:
            run([sys.executable, MAPS / "trainworld" / "tools" / "snap_wasm.py", MAPS / ".tw-snap-track-target", MAPS / ".tw-snap-track-target-demand"], APP, env)
        finally:
            subprocess.run([sys.executable, str(MAPS / "cores.py"), "release", "snaptrack"], cwd=str(MAPS), creationflags=NO_WINDOW)
    for d in ("dist-new", "dist-old"):
        shutil.rmtree(APP / d, ignore_errors=True)
    run(["npx", "tsc", "--noEmit"], APP, env)
    run(["npx", "vite", "build", "--outDir", "dist-new"], APP, env)
    if not (APP / "dist-new" / "index.html").exists():
        sys.exit("vite wrote no index.html")
    # the packs in a build are hard links to data/packs, so deleting the old build leaves the data alone
    if (APP / "dist").exists():
        (APP / "dist").rename(APP / "dist-old")
    (APP / "dist-new").rename(APP / "dist")
    shutil.rmtree(APP / "dist-old", ignore_errors=True)
    served = subprocess.run([sys.executable, str(MAPS / "serve.py"), "list"], cwd=str(MAPS), capture_output=True, text=True, creationflags=NO_WINDOW).stdout
    if not re.search(rf"^{NAME}\s", served, re.M):
        run([sys.executable, MAPS / "serve.py", "add", NAME, APP / "dist", "track branch build (open with trainworld/?track)"], MAPS)
        run([sys.executable, MAPS / "serve.py", "hide", NAME], MAPS)
    print(f"built {built[:8]}{' with local changes' if as_is and git('status', '--porcelain', cwd=TREE) else ''}: http://localhost:8800/trainworld/?track")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
