"""T-1065: on the rented machine, one runner region's results as a tar for the bucket (wave.sh's
`send`). docs/aws.md; land.py unpacks it at home. Runs from the machine's /opt/tw.

    wavepack.py <name> <id,id,...> <out.tgz> [<t0 marker file>]

Takes, for each row, whatever is on disk (a spot notice packs a region mid-way): data/packs/<id>.*,
data/work/<id>.*, data/work/<folder>/<id>.* (res9, pre-T509s4, ...), the row's OSM caches in
data/raw (tier0/, t142b/, t196/), the state caches data/work/usa/blocks_<fips>.npz of its counties,
the batch logs of the runs that built it, and the region's run and read logs. manifest.json holds
the machine's plan row of each id, and every file's size and SHA-256; newer.txt every other file
under data/ written since the marker (what this list may have missed).
"""
import glob
import hashlib
import io
import json
import os
import platform
import re
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TW = HERE.parents[1]
BASE = TW.parent          # /opt/tw: trainworld/, artifacts/, logs/
SKIP_WORK = {"usa", "target", "world_run", "res9"}


def row_files(cid, row):
    d = TW / "data"
    out = set(glob.glob(str(d / "packs" / f"{cid}.*")))
    out |= set(glob.glob(str(d / "work" / f"{cid}.*")))
    out |= set(glob.glob(str(d / "work" / "res9" / f"{cid}.*")))
    for sub in (d / "work").iterdir() if (d / "work").is_dir() else []:
        if sub.is_dir() and sub.name not in SKIP_WORK:
            out |= set(glob.glob(str(sub / f"{cid}.*")))
    pat = re.compile(rf"_{re.escape(cid)}\.(json|csv\.gz)$")
    for sub in ("tier0", "t142b", "t196"):
        p = d / "raw" / sub
        if p.is_dir():
            out |= {str(f) for f in p.iterdir() if pat.search(f.name)}
    out |= set(glob.glob(str(d / "raw" / f"osm_places_{cid}.json")))  # an Overpass answer: should not exist
    states = sorted({u[:2] for u in (row.get("units") or {}).get("USA", [])})
    out |= {str(d / "work" / "usa" / f"blocks_{s}.npz") for s in states if (d / "work" / "usa" / f"blocks_{s}.npz").exists()}
    return sorted(p for p in out if os.path.isfile(p) and not p.endswith(".part"))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(name, ids, out, marker=None):
    ids = [i for i in ids.split(",") if i]
    plan = json.loads((TW / "pipeline" / "world_plan.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in plan["metros"] if r.get("units") and r["id"] in ids}
    files, by_row = set(), {}
    for cid in ids:
        fs = row_files(cid, rows.get(cid, {}))
        by_row[cid] = [os.path.relpath(p, BASE).replace(os.sep, "/") for p in fs]
        files |= set(fs)
    batches = sorted({(r.get("run") or {}).get("run_batch") for r in rows.values()} - {None})
    for bt in batches:
        files |= {str(p) for p in (BASE / "artifacts/trainworld/world_run" / bt).rglob("*") if p.is_file()}
    files |= {str(p) for p in (BASE / "logs").glob("*") if p.is_file() and (p.name == f"run_{name}.log" or p.name.startswith("read_") or p.name == "mem.txt")}
    rel = {p: os.path.relpath(p, BASE).replace(os.sep, "/") for p in files}
    newer = []
    if marker and os.path.exists(marker):
        t0 = os.path.getmtime(marker)
        for top, dirs, fs in os.walk(TW / "data"):
            dirs[:] = [x for x in dirs if not (Path(top) / x).match("*/raw/osm") and x != "target"]
            for f in fs:
                p = os.path.join(top, f)
                if p not in files and os.path.getmtime(p) > t0:
                    newer.append(f"{os.path.getsize(p):>12} {os.path.relpath(p, BASE)}")
    manifest = {"name": name, "ids": ids, "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "machine": platform.platform(), "python": platform.python_version(), "batches": batches,
                "rows": rows, "row_files": by_row,
                "files": {rel[p]: {"size": os.path.getsize(p), "sha256": sha(p)} for p in sorted(files)}}
    part = f"{out}.{os.getpid()}.part"
    with tarfile.open(part, "w:gz", compresslevel=3) as t:
        for name_, text in (("manifest.json", json.dumps(manifest, indent=1)), ("newer.txt", "\n".join(sorted(newer)) + "\n")):
            data = text.encode("utf-8")
            ti = tarfile.TarInfo(name_)
            ti.size, ti.mtime = len(data), int(time.time())
            t.addfile(ti, io.BytesIO(data))
        for p in sorted(files):
            t.add(p, arcname=rel[p], recursive=False)
    os.replace(part, out)
    built = [i for i in ids if (rows.get(i) or {}).get("status") == "built"]
    print(f"{name}: {len(files)} files, {os.path.getsize(out) / 1e6:.1f} MB; built {len(built)} of {len(ids)}"
          f"{'; not built: ' + ' '.join(i for i in ids if i not in built) if len(built) < len(ids) else ''}"
          f"{'; ' + str(len(newer)) + ' other new files under data/ (newer.txt)' if newer else ''}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
