"""T-928: the published data's last good state, frozen, so a publish never waits for a batch.

    python -I pipeline/freeze.py [--label L] [--force] [--root DIR]   freeze the data now
    python -I pipeline/freeze.py list [--root DIR]                    the freezes, newest first
    python -I pipeline/freeze.py check [stamp] [--root DIR]           every frozen file still as frozen
    python -I pipeline/freeze.py build [stamp] [--tree DIR] [--out DIR] [--root DIR]
                                                                      the tree's app built on the freeze

A freeze is `data/frozen/<stamp>/`: `packs/` holds hard links to every file a production build
copies from data/packs (vite.config.ts: each registered zone's five files, the market pack,
outlines.json, realms.geojson, real/*.save, realgraph/*.rgr and its index.json), `game/` copies of the four data
files the app bundles (registry.json, market-rows.json, zone-aliases.json, zone-names.json), and
`manifest.json` each file's size and modified time. A hard link keeps a file as it was because every
pipeline writer replaces a file (a .part renamed over it) and none rewrites one in place; `check`
would show a writer that did. Links need the volume data/packs is on (NTFS: no space, seconds).

The runner freezes at the end of every run whose join ended clean, holding the packs lock
(`--locked`). By hand, a freeze takes the packs and registry locks, waits for gates under way and
refuses while a run is live, unless --force. The newest KEEP freezes stay. A registered zone
missing a file stops the freeze (it would not be a good state).

`build`: in the tree (default maps/.tw-snap, the snapshot of a commit) the freeze's four game files
are laid over the tree's for the build only (`git checkout` puts the tree's back), then Vite builds
with TW_PACKS=<freeze>/packs into --out (default artifacts/trainworld/publish/<stamp>/dist), which
gets `frozen.json` naming the freeze and the tree; `tools/deploy.py --dist <out>` publishes that.
The tree's wasm must be built already (its last `npm run build` made it).
"""
import datetime
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import shared_lock  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT.parent
KEEP = 3
GAME = ("registry.json", "market-rows.json", "zone-aliases.json", "zone-names.json")
ZONE_FILES = (".json", ".bin", ".boundary.geojson", ".places.json", ".market.json")


def frozen_dir(root):
    return Path(root) / "data" / "frozen"


def build_files(root):
    """Every data/packs file a production build copies (vite.config.ts), relative; and the
    registered zones' files that are missing."""
    packs = Path(root) / "data/packs"
    reg = shared_lock.read_json(Path(root) / "app/src/game/registry.json")
    want = [c["id"] + x for c in reg["cities"] for x in ZONE_FILES] + [reg["market"]["pack"] + ".json", "outlines.json"]
    have = [f for f in want if (packs / f).is_file()]
    if (packs / "realms.geojson").is_file():  # T-975: optional, as the app shows every zone without it
        have.append("realms.geojson")
    missing = [f for f in want if f not in have]
    for sub, ok in (("real", lambda n: n.endswith(".save")), ("realgraph", lambda n: n.endswith(".rgr") or n == "index.json")):
        if (packs / sub).is_dir():
            have += [f"{sub}/{n}" for n in sorted(os.listdir(packs / sub)) if ok(n)]
    return have, missing


def live_runs(root):
    """Runs alive now (their batch locks held), by name."""
    locks = Path(root) / "data/work/world_run/locks"
    return [p.stem for p in locks.glob("batch-*.lock") if shared_lock.is_held(p.stem, root)] if locks.is_dir() else []


def freeze(root=ROOT, label="", force=False, locked=False):
    root = Path(root)
    if not locked:
        runs = live_runs(root)
        if runs and not force:
            sys.exit(f"not frozen: {', '.join(runs)} live (a run's rows may be registered and not yet joined); --force to freeze anyway")
        with shared_lock.held("packs", "freeze.py", root=root):
            locks = root / "data/work/world_run/locks"
            while any(shared_lock.is_held(p.stem, root) for p in locks.glob("gate-*.lock")):
                time.sleep(5)  # a gate under way registers packs it may yet park
            with shared_lock.held("registry", "freeze.py", root=root):
                return _freeze(root, label, force)
    with shared_lock.held("registry", "freeze.py", root=root):
        return _freeze(root, label, force)


def _freeze(root, label, force):
    t0 = time.time()
    files, missing = build_files(root)
    if missing and not force:
        sys.exit(f"not frozen: {len(missing)} files the build copies are missing: {', '.join(missing[:8])}")
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = frozen_dir(root)
    tmp = base / f"{stamp}.part"
    (tmp / "packs").mkdir(parents=True)
    (tmp / "game").mkdir()
    packs = root / "data/packs"
    man = {"stamp": stamp, "label": label, "made": datetime.datetime.now().isoformat(timespec="seconds"),
           "missing": missing, "files": {}}
    for rel in files:
        src, dst = packs / rel, tmp / "packs" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.link(src, dst)
        st = dst.stat()
        man["files"]["packs/" + rel] = [st.st_size, st.st_mtime_ns]
    for g in GAME:
        src = root / "app/src/game" / g
        if src.exists():
            shutil.copy2(src, tmp / "game" / g)
            st = (tmp / "game" / g).stat()
            man["files"]["game/" + g] = [st.st_size, st.st_mtime_ns]
    man["zones"] = len(json.loads((tmp / "game/registry.json").read_text(encoding="utf-8"))["cities"])
    (tmp / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    os.replace(tmp, base / stamp)
    gone = []
    for old in sorted((p for p in base.iterdir() if p.is_dir() and not p.name.endswith(".part")), reverse=True)[KEEP:]:
        shutil.rmtree(old)
        gone.append(old.name)
    print(f"froze {len(files)} pack files and {sum(1 for k in man['files'] if k.startswith('game/'))} game files of "
          f"{man['zones']} zones as {base / stamp} in {time.time() - t0:.1f} s"
          f"{'; ' + str(len(missing)) + ' missing' if missing else ''}{'; removed ' + ' '.join(gone) if gone else ''}", flush=True)
    return base / stamp


def freezes(root=ROOT):
    base = frozen_dir(root)
    return sorted((p for p in base.iterdir() if p.is_dir() and not p.name.endswith(".part")), reverse=True) if base.is_dir() else []


def pick(root, stamp=None):
    fs = freezes(root)
    if not fs:
        sys.exit(f"no freezes in {frozen_dir(root)}")
    if stamp is None:
        return fs[0]
    for p in fs:
        if p.name == stamp:
            return p
    sys.exit(f"no freeze {stamp} in {frozen_dir(root)}")


def check(path):
    """[problems]: frozen files changed or gone since the freeze."""
    man = json.loads((Path(path) / "manifest.json").read_text(encoding="utf-8"))
    bad = []
    for rel, (size, mt) in man["files"].items():
        p = Path(path) / rel
        if not p.exists():
            bad.append(f"{rel} gone")
            continue
        st = p.stat()
        if st.st_size != size or st.st_mtime_ns != mt:
            bad.append(f"{rel} changed (a writer rewrote the file in place, so the live file changed too)")
    return bad


def build(path, tree, out):
    """The tree's app built on the freeze's data, into `out`."""
    path, tree, out = Path(path), Path(tree), Path(out)
    app = tree / "trainworld/app"
    bad = check(path)
    if bad:
        sys.exit(f"freeze {path.name} is not as frozen: {'; '.join(bad[:5])}")
    rels = [f"trainworld/app/src/game/{g}" for g in GAME if (path / "game" / g).exists()]
    dirty = subprocess.run(["git", "-C", str(tree), "status", "--porcelain", "--", *rels], capture_output=True, text=True).stdout.strip()
    if dirty:
        sys.exit(f"{tree} has changes in the game data files, which a build on a freeze replaces for a while:\n{dirty}")
    vite = app / "node_modules/vite/bin/vite.js"
    if not vite.exists() or not (app / "src/wasm/track").is_dir():
        sys.exit(f"{app} lacks node_modules or its wasm: run its `npm run build` once first")
    commit = subprocess.run(["git", "-C", str(tree), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    t0 = time.time()
    try:
        for g in GAME:
            if (path / "game" / g).exists():
                shutil.copyfile(path / "game" / g, app / "src/game" / g)
        p = subprocess.run(["node", str(vite), "build", "--outDir", str(out), "--emptyOutDir"], cwd=str(app),
                           env=dict(os.environ, TW_PACKS=str(path / "packs")))
    finally:
        subprocess.run(["git", "-C", str(tree), "checkout", "--", *rels], check=True)
    if p.returncode:
        sys.exit(f"vite build failed ({p.returncode})")
    bad = check(path)
    if bad:
        sys.exit(f"freeze {path.name} changed during the build: {'; '.join(bad[:5])}")
    man = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    (out / "frozen.json").write_text(json.dumps({"task": "T-928", "freeze": str(path), "stamp": man["stamp"], "label": man["label"],
                                                 "zones": man["zones"], "tree": str(tree), "app": str(app), "commit": commit,
                                                 "built": datetime.datetime.now().isoformat(timespec="seconds")}, indent=1), encoding="utf-8")
    print(f"built {commit} of {tree} on freeze {man['stamp']} ({man['label'] or 'no label'}, {man['zones']} zones) into {out} "
          f"in {time.time() - t0:.0f} s", flush=True)


def main(argv):
    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default
    root = Path(opt("--root", ROOT)).resolve()
    valued = {i for i, a in enumerate(argv) if i and argv[i - 1] in ("--root", "--label", "--tree", "--out")}
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and i not in valued]
    cmd = pos[0] if pos else "freeze"
    if cmd == "freeze":
        freeze(root, opt("--label", ""), "--force" in argv, "--locked" in argv)
    elif cmd == "list":
        for p in freezes(root):
            man = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
            print(f"{p.name}  {man['label'] or '-':24} {man['zones']} zones, {len(man['files'])} files{'  ' + str(len(check(p))) + ' changed' if check(p) else ''}")
    elif cmd == "check":
        p = pick(root, pos[1] if len(pos) > 1 else None)
        bad = check(p)
        print(f"{p.name}: " + ("as frozen" if not bad else f"{len(bad)} changed: " + "; ".join(bad[:10])))
        sys.exit(1 if bad else 0)
    elif cmd == "build":
        p = pick(root, pos[1] if len(pos) > 1 else None)
        build(p, Path(opt("--tree", MAPS / ".tw-snap")), Path(opt("--out", MAPS / "artifacts/trainworld/publish" / p.name / "dist")))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main(sys.argv[1:])
