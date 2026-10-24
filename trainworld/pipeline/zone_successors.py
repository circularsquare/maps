"""T-1053: the successor of every zone id the registry has dropped, in app/src/game/zone-successors.json.

A save names zones by id (its cities, start and opened zones, region ledgers, stars); a save naming
an id a replan removed loads with the id taken as its successor (workers/saveFile.ts `zoneNow`), or
dropped where the table says null. The successor is the zone whose outline (data/packs/outlines.json,
so run this after the outlines are rebuilt) holds the removed zone's old centre, else the zone with an
outline point nearest it within 30 km, else null. Entries are only added (a hand `--set` replaces
one); an id that comes back is the registry's again, since the app looks there first.

    python pipeline/zone_successors.py                 ids in HEAD's registry the working one lacks
    python pipeline/zone_successors.py --history       every id any commit's registry had
    python pipeline/zone_successors.py --set wnd=nlc   a hand choice (`wnd=` for none)
    python pipeline/zone_successors.py --dry           print, write nothing

Smoke's t1053-zone-ids check fails while a removed id has no entry.
"""
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAPS = ROOT.parent
REG_REL = "trainworld/app/src/game/registry.json"
NEAR_KM = 30.0


def git_registry(rev):
    out = subprocess.run(["git", "-C", str(MAPS), "show", f"{rev}:{REG_REL}"], capture_output=True)
    if out.returncode:
        return None
    return json.loads(out.stdout.decode("utf-8"))


def old_rows(history):
    """id -> its last registry row, from HEAD or (history) every commit that changed the registry."""
    revs = ["HEAD"]
    if history:
        log = subprocess.run(["git", "-C", str(MAPS), "log", "--format=%H", "--", REG_REL], capture_output=True, text=True, check=True)
        revs = log.stdout.split()
    rows = {}
    for i, rev in enumerate(revs):
        reg = git_registry(rev)
        if history and i % 25 == 0:
            print(f"  {i}/{len(revs)} commits", file=sys.stderr)
        for c in (reg or {}).get("cities", []):
            rows.setdefault(c["id"], c)
    return rows


def decode(flat, q):
    x, y = flat[0], flat[1]
    pts = [(x / q, y / q)]
    for i in range(2, len(flat) - 1, 2):
        x += flat[i]
        y += flat[i + 1]
        pts.append((x / q, y / q))
    return pts


def load_outlines(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for zid, rings in doc["zones"].items():
        rs = [decode(r, doc["q"]) for r in rings]
        xs = [p[0] for r in rs for p in r]
        ys = [p[1] for r in rs for p in r]
        if xs:
            out[zid] = (rs, (min(xs), min(ys), max(xs), max(ys)))
    return out


def inside(rings, x, y):
    hit = False
    for r in rings:
        j = len(r) - 1
        for i in range(len(r)):
            (xi, yi), (xj, yj) = r[i], r[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                hit = not hit
            j = i
    return hit


def km(a, b):
    kx = 111.32 * math.cos(math.radians((a[1] + b[1]) / 2))
    return math.hypot((a[0] - b[0]) * kx, (a[1] - b[1]) * 110.57)


def successor(centre, outlines, have):
    x, y = centre
    for zid, (rings, (x0, y0, x1, y1)) in outlines.items():
        if zid in have and x0 <= x <= x1 and y0 <= y <= y1 and inside(rings, x, y):
            return zid
    pad = NEAR_KM / 80.0
    best, best_km = None, NEAR_KM
    for zid, (rings, (x0, y0, x1, y1)) in outlines.items():
        if zid not in have or x < x0 - pad * 2 or x > x1 + pad * 2 or y < y0 - pad or y > y1 + pad:
            continue
        d = min(km(centre, p) for r in rings for p in r)
        if d < best_km:
            best, best_km = zid, d
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="ID=SUCCESSOR")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--registry", default=str(ROOT / "app/src/game/registry.json"))
    ap.add_argument("--outlines", default=str(ROOT / "data/packs/outlines.json"))
    ap.add_argument("--out", default=str(ROOT / "app/src/game/zone-successors.json"))
    a = ap.parse_args()

    have = {c["id"] for c in json.loads(Path(a.registry).read_text(encoding="utf-8"))["cities"]}
    out_path = Path(a.out)
    table = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    added = {}
    for s in a.set:
        zid, _, to = s.partition("=")
        if to and to not in have:
            sys.exit(f"--set {s}: {to} is not registered")
        added[zid] = to or None
    rows = old_rows(a.history)
    gone = sorted(z for z in rows if z not in have and z not in table and z not in added)
    if gone:
        outlines = load_outlines(a.outlines)
        for zid in gone:
            added[zid] = successor(rows[zid].get("center") or [rows[zid]["origin"]["lon"], rows[zid]["origin"]["lat"]], outlines, have)
    for zid, to in sorted(added.items()):
        print(f"{zid} ({rows.get(zid, {}).get('name', '?')}) -> {to}")
    print(f"RESULT {len(added)} added, {len(table) + len([z for z in added if z not in table])} in the table")
    if added and not a.dry:
        table.update(added)
        tmp = out_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(sorted(table.items())), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(out_path)


if __name__ == "__main__":
    main()
