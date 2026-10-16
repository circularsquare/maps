"""Scan a drive and write data.js for the diskmap treemap page (index.html).

    python diskmap/scan.py            # scans C:\
    python diskmap/scan.py D:\        # another drive or folder

Sizes are logical file sizes. Junctions and symlinks are not followed, so nothing is
counted twice through them. OneDrive files that live only in the cloud take no disk and
are left out of the boxes (their total is reported separately). Folders Windows will not
let us read show up as the "not readable" gap between the drive's used space and the scan.
"""
import heapq
import json
import os
import shutil
import stat
import sys
import time

KEEP_BYTES = 150 * 2**20   # a folder or file smaller than this is folded into "smaller items"
TOP_FILES = 300

REPARSE = stat.FILE_ATTRIBUTE_REPARSE_POINT
CLOUD_ONLY = 0x00400000 | 0x00040000 | 0x00001000  # recall on data access / on open, offline

HERE = os.path.dirname(os.path.abspath(__file__))


class Scan:
    def __init__(self):
        self.dirs = 0
        self.files = 0
        self.denied = 0
        self.cloud_bytes = 0
        self.big = []  # min-heap of (size, path, mtime)
        self.t0 = time.time()
        self.last = 0

    def walk(self, path, name):
        """Return node dict {n, s, f, m, a, c?} for a directory.

        m is the newest change anywhere inside; a is the size-weighted average change time,
        so a folder of old data with one fresh file still reads as old.
        """
        self.dirs += 1
        now = time.time()
        if now - self.last > 5:
            self.last = now
            print(f"  {now - self.t0:5.0f}s  {self.dirs:,} folders  {self.files:,} files  {path[:90]}",
                  flush=True)
        size = 0
        nfiles = 0
        newest = 0.0
        wsum = 0.0  # sum of size * mtime
        kids = []
        small_s = 0
        small_f = 0
        small_n = 0
        small_w = 0.0
        try:
            it = os.scandir(path)
        except OSError:
            self.denied += 1
            return {"n": name, "s": 0, "f": 0, "m": 0, "a": 0, "x": 1}
        with it:
            for e in it:
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                attrs = getattr(st, "st_file_attributes", 0)
                if attrs & REPARSE:
                    continue  # junction, symlink, or other reparse point: don't follow or count
                if stat.S_ISDIR(st.st_mode):
                    child = self.walk(e.path, e.name)
                    size += child["s"]
                    nfiles += child["f"]
                    newest = max(newest, child["m"])
                    wsum += child["s"] * child["a"]
                    if child["s"] >= KEEP_BYTES:
                        kids.append(child)
                    else:
                        small_s += child["s"]
                        small_f += child["f"]
                        small_w += child["s"] * child["a"]
                        small_n += 1
                else:
                    if attrs & CLOUD_ONLY:
                        self.cloud_bytes += st.st_size
                        continue
                    s = st.st_size
                    self.files += 1
                    size += s
                    nfiles += 1
                    newest = max(newest, st.st_mtime)
                    wsum += s * st.st_mtime
                    if len(self.big) < TOP_FILES:
                        heapq.heappush(self.big, (s, e.path, st.st_mtime))
                    elif s > self.big[0][0]:
                        heapq.heapreplace(self.big, (s, e.path, st.st_mtime))
                    if s >= KEEP_BYTES:
                        kids.append({"n": e.name, "s": s, "f": 1, "m": int(st.st_mtime),
                                     "a": int(st.st_mtime), "file": 1})
                    else:
                        small_s += s
                        small_f += 1
                        small_w += s * st.st_mtime
                        small_n += 1
        if kids and small_s:
            kids.append({"n": f"{small_n:,} smaller items", "s": small_s, "f": small_f,
                         "m": 0, "a": int(small_w / small_s), "rest": 1})
        node = {"n": name, "s": size, "f": nfiles, "m": int(newest),
                "a": int(wsum / size) if size else 0}
        if kids:
            node["c"] = sorted(kids, key=lambda k: -k["s"])
        return node


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "C:\\"
    root = os.path.abspath(root)
    sys.setrecursionlimit(20000)
    print(f"Scanning {root} ...", flush=True)
    sc = Scan()
    tree = sc.walk(root, root)
    du = shutil.disk_usage(root)
    gap = du.used - tree["s"]
    if gap > KEEP_BYTES:
        tree.setdefault("c", []).append(
            {"n": "not readable (system, other users, hard links)", "s": gap, "f": 0, "m": 0, "a": 0, "gap": 1})
        tree["c"].sort(key=lambda k: -k["s"])
        tree["s"] += gap
    out = {
        "root": root,
        "scanned": int(time.time()),
        "seconds": round(time.time() - sc.t0),
        "disk": {"total": du.total, "used": du.used, "free": du.free},
        "scannedBytes": tree["s"] - max(gap, 0),
        "cloudOnly": sc.cloud_bytes,
        "denied": sc.denied,
        "files": sc.files,
        "dirs": sc.dirs,
        "tree": tree,
        "big": [{"p": p, "s": s, "m": int(m)} for s, p, m in sorted(sc.big, reverse=True)],
    }
    dest = os.path.join(HERE, "data.js")
    tmp = dest + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("window.DISK = ")
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        f.write(";\n")
    os.replace(tmp, dest)
    gb = 2**30
    print(f"Done in {out['seconds']}s: {sc.files:,} files, {sc.dirs:,} folders, "
          f"{tree['s'] / gb:.1f} GB in the diagram, {gap / gb:.1f} GB not readable, "
          f"{sc.cloud_bytes / gb:.1f} GB cloud-only, {sc.denied:,} folders refused. Wrote {dest}")


if __name__ == "__main__":
    main()
