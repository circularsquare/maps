"""T-2962: keep the previous build's hashed files beside the new build's.

A tab opened before a publish (or before 8800's rebuild) still asks for the old build's lazy files: a
language's messages, a worker. So a build's folder keeps the files of the build before it, and drops the
ones older than that. Which files belong to a build is `build-files.json` in the folder's root, written
each time (a sorted list of the paths under assets/ and fonts/ the build made). A folder with no manifest
(first run) keeps everything it has as the previous build.

    python trainworld/tools/keep_assets.py carry <old dist> <new dist>   before a dist swap: the new dist
        gets the old dist's previous-build files, and its own manifest
    keep_assets.carry_into(dest, dist)                                  deploy.py: dist's files into the
        website folder, the folder's older files removed
"""

import json
import os
import shutil
import sys

MANIFEST = "build-files.json"
DIRS = ("assets", "fonts")


def build_files(root):
    """The build's own hashed files: from its manifest when it has one (a dist that already carries a
    previous build's files lists only its own), else everything under assets/ and fonts/."""
    m = os.path.join(root, MANIFEST)
    if os.path.exists(m):
        return set(json.load(open(m, encoding="utf-8")))
    return present(root)


def present(root):
    out = set()
    for d in DIRS:
        for r, _, fs in os.walk(os.path.join(root, d)):
            for f in fs:
                out.add(os.path.relpath(os.path.join(r, f), root).replace(os.sep, "/"))
    return out


def write_manifest(root, files):
    with open(os.path.join(root, MANIFEST), "w", encoding="utf-8", newline="\n") as f:
        json.dump(sorted(files), f, indent=0)


def _copy(src_root, dst_root, rel):
    t = os.path.join(dst_root, *rel.split("/"))
    os.makedirs(os.path.dirname(t), exist_ok=True)
    shutil.copy2(os.path.join(src_root, *rel.split("/")), t)


def carry(old, new):
    """Into the new dist (not yet swapped in): the old dist's own files that the new one lacks. The new
    dist's manifest lists only what the new build made."""
    mine = present(new)
    write_manifest(new, mine)
    if not os.path.isdir(old):
        return 0
    n = 0
    for rel in build_files(old) - mine:
        if os.path.exists(os.path.join(old, *rel.split("/"))):
            _copy(old, new, rel)
            n += 1
    return n


def carry_into(dest, dist, dry=False):
    """dist's files go into dest (a website folder holding an earlier publish): dest's files stay if they
    belong to the build it holds now, and go if older. Returns (kept, removed)."""
    mine = build_files(dist)
    held = build_files(dest) if os.path.isdir(dest) else set()
    gone = present(dest) - held - mine if os.path.isdir(dest) else set()
    if not dry:
        for rel in gone:
            os.remove(os.path.join(dest, *rel.split("/")))
        for rel in sorted(mine):
            _copy(dist, dest, rel)
        write_manifest(dest, mine)
    return len((held - mine) & present(dest)) if os.path.isdir(dest) else 0, len(gone)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "carry":
        print(f"carried {carry(sys.argv[2], sys.argv[3])} files from the previous build")
    else:
        sys.exit(__doc__)
