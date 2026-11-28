"""Sweep old session scratch under %TEMP%\\claude (T-4096).

Layout: claude/<project hash>/<session folder>/..., and claude/<other>/<entry>.
The unit is a folder two levels down (a top-level file counts as a unit too).
A unit goes when its newest file or folder is more than --days old (default 2).

    python trainworld/tools/scratch_sweep.py            # dry run: lists, deletes nothing
    python trainworld/tools/scratch_sweep.py --yes      # deletes
    --days N     age limit        --keep PATH   never touch this folder (repeatable)
    --root PATH  sweep another folder (for testing)

Never follows a junction or symlink (they are unlinked, not entered). A file a
running session holds cannot be removed: it is left, and its folder stays.
This script is run by hand; it is not a scheduled task.
"""
import argparse
import os
import stat
import sys
import time

REPARSE = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def long(p):
    p = os.path.abspath(p)
    if os.name == "nt" and not p.startswith("\\\\?\\"):
        p = "\\\\?\\" + p
    return p


def is_link(st):
    return stat.S_ISLNK(st.st_mode) or bool(getattr(st, "st_file_attributes", 0) & REPARSE)


def measure(path):
    """(newest mtime, bytes, file count) without entering links."""
    try:
        st = os.lstat(long(path))
    except OSError:
        return 0.0, 0, 0
    newest, size, n = st.st_mtime, 0, 0
    if is_link(st) or not stat.S_ISDIR(st.st_mode):
        return newest, (0 if is_link(st) else st.st_size), 1
    stack = [path]
    while stack:
        d = stack.pop()
        try:
            it = list(os.scandir(long(d)))
        except OSError:
            continue
        for e in it:
            try:
                s = e.stat(follow_symlinks=False)
            except OSError:
                continue
            newest = max(newest, s.st_mtime)
            n += 1
            if is_link(s):
                continue
            if stat.S_ISDIR(s.st_mode):
                stack.append(os.path.join(d, e.name))
            else:
                size += s.st_size
    return newest, size, n


def remove(path):
    """Delete a unit; returns the list of paths that could not be removed."""
    failed = []

    def rm(p):
        try:
            st = os.lstat(long(p))
        except OSError:
            return
        if is_link(st):
            try:
                os.rmdir(long(p)) if stat.S_ISDIR(st.st_mode) else os.unlink(long(p))
            except OSError:
                failed.append(p)
            return
        if stat.S_ISDIR(st.st_mode):
            try:
                names = os.listdir(long(p))
            except OSError:
                failed.append(p)
                return
            for n in names:
                rm(os.path.join(p, n))
            try:
                os.rmdir(long(p))
            except OSError:
                failed.append(p)
            return
        try:
            os.chmod(long(p), stat.S_IWRITE)
            os.unlink(long(p))
        except OSError:
            failed.append(p)

    rm(path)
    return failed


def units(root):
    for top in sorted(os.listdir(root)):
        tp = os.path.join(root, top)
        try:
            st = os.lstat(long(tp))
        except OSError:
            continue
        if is_link(st) or not stat.S_ISDIR(st.st_mode):
            yield tp
            continue
        for sub in sorted(os.listdir(long(tp))):
            yield os.path.join(tp, sub)


def gb(b):
    return "%.2f GB" % (b / 1e9) if b >= 1e8 else "%.1f MB" % (b / 1e6)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="delete (default is a dry run)")
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--keep", action="append", default=[])
    ap.add_argument("--root", default=os.path.join(os.environ.get("TEMP", ""), "claude"))
    a = ap.parse_args()
    root = os.path.abspath(a.root)
    if not os.path.isdir(root):
        sys.exit("no folder: " + root)
    keep = [os.path.normcase(os.path.abspath(k)) for k in a.keep]
    cutoff = time.time() - a.days * 86400
    old = fresh = kept = 0
    freed = 0
    left = []
    for u in units(root):
        nu = os.path.normcase(os.path.abspath(u))
        if any(k == nu or k.startswith(nu + os.sep) or nu.startswith(k + os.sep) for k in keep):
            kept += 1
            print("keep   (--keep)      %s" % u)
            continue
        newest, size, n = measure(u)
        age = (time.time() - newest) / 86400
        if newest > cutoff:
            fresh += 1
            continue
        old += 1
        print("%-6s %9s %5.1f d  %s" % ("delete" if a.yes else "old", gb(size), age, u))
        if a.yes:
            bad = remove(u)
            if bad:
                left.append((u, len(bad)))
            else:
                freed += size
    print("%d old, %d newer than %g days (left alone), %d kept" % (old, fresh, a.days, kept))
    if a.yes:
        print("deleted %s; %d folders held by a session were left" % (gb(freed), len(left)))
        for u, c in left:
            print("  held: %s (%d items)" % (u, c))
    else:
        print("dry run: nothing deleted. add --yes to delete")


if __name__ == "__main__":
    main()
