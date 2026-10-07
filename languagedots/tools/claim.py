"""Who is working on which country. Advisory: it cannot stop anyone, it lets agents see each other.

    python tools/claim.py                              what is claimed, parked, waiting, free
    python tools/claim.py take <cc> --id <sid> [--note "..."]
    python tools/claim.py park <cc> --id <sid>         stop cleanly; writes handoff/<cc>.md
    python tools/claim.py done <cc> --id <sid>         checks the country is built, marks it drawn
    python tools/claim.py drop <cc> --id <sid>         release without parking (you did nothing)

State: claims.json (who holds what) and queue.csv's `status` column (free / parked / drawn /
ruling / blocked / closed). Both are edited under one lock, because several agents run this at
once. A country is WAITING FOR THE BUILD TAIL when its dots file is newer than the last tile build.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "queue.csv"
CLAIMS = ROOT / "claims.json"
LOCK = ROOT / "claims.lock"
HANDOFF = ROOT / "handoff"
PROC = ROOT / "data" / "processed"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class Locked:
    def __enter__(self):
        for _ in range(300):
            try:
                os.close(os.open(LOCK, os.O_CREAT | os.O_EXCL))
                return self
            except FileExistsError:
                try:
                    if time.time() - LOCK.stat().st_mtime > 60:
                        LOCK.unlink(missing_ok=True)
                except FileNotFoundError:
                    pass
                time.sleep(0.1)
        raise SystemExit(f"{LOCK} held for 30 s; delete it if nothing is running claim.py")

    def __exit__(self, *a):
        LOCK.unlink(missing_ok=True)


def read_queue():
    with open(QUEUE, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write_queue(fields, rows):
    tmp = QUEUE.with_suffix(".csv.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, QUEUE)


def read_claims():
    try:
        return json.loads(CLAIMS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_claims(c):
    tmp = CLAIMS.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(c, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, CLAIMS)


def set_status(cc, status, note=None):
    fields, rows = read_queue()
    hit = False
    for r in rows:
        if r["cc"] == cc:
            r["status"] = status
            if note is not None:
                r["note"] = note
            hit = True
    if not hit:
        rows.append({k: "" for k in fields} | {"cc": cc, "status": status, "note": note or ""})
    write_queue(fields, rows)


def last_build():
    p = PROC / "build_tail.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("finished", 0)
    except (OSError, ValueError):
        return 0


def show():
    _, rows = read_queue()
    claims = read_claims()
    built_at = last_build()
    print("CLAIMED")
    for cc, c in claims.items():
        print(f"  {cc:<6} {c['id']:<40} {time.strftime('%m-%d %H:%M', time.localtime(c['since']))}  {c.get('note', '')}")
    if not claims:
        print("  (none)")
    parked = sorted(p.stem for p in HANDOFF.glob("*.md")) if HANDOFF.exists() else []
    print("PARKED (take these first; handoff/<cc>.md says where they stopped)")
    print("  " + (" ".join(parked) if parked else "(none)"))
    waiting = [p.stem[5:] for p in PROC.glob("dots_*.geojson")
               if p.stat().st_mtime > built_at and "_" not in p.stem[5:]]
    print("WAITING FOR THE BUILD TAIL (dots newer than the last tiles.py)")
    print("  " + (" ".join(sorted(waiting)) if waiting else "(none)")
          + ("" if built_at else "   [no build_tail.json: last build time unknown]"))
    free = [r for r in rows if r["status"] == "free" and r["cc"] not in claims and r["cc"] not in parked]
    print(f"FREE, in queue order ({len(free)}; tier, millions, religiondots geography)")
    for r in free[:25]:
        print(f"  {r['cc']:<6} {r['tier']} {float(r['pop_m'] or 0):>7.1f}m  {'rd-geo' if r['rd_geo'] else '      '}  "
              f"{r['country'][:28]:<28} {r['question']:<16} {r['finest_level'][:40]}")
    if len(free) > 25:
        print(f"  … {len(free) - 25} more in queue.csv")
    n = {}
    for r in rows:
        n[r["status"]] = n.get(r["status"], 0) + 1
    print("STATUS COUNTS", n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="show", choices=["show", "take", "park", "done", "drop"])
    ap.add_argument("cc", nargs="?")
    ap.add_argument("--id")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.cmd == "show":
        return show()
    if not a.cc or not a.id:
        raise SystemExit("take/park/done/drop need <cc> and --id <sid>")
    cc = a.cc.lower()
    with Locked():
        claims = read_claims()
        held = claims.get(cc)
        if a.cmd == "take":
            if held and held["id"] != a.id:
                raise SystemExit(f"{cc} is held by {held['id']} since "
                                 f"{time.strftime('%H:%M', time.localtime(held['since']))}: pick another")
            claims[cc] = {"id": a.id, "since": time.time(), "note": a.note}
            write_claims(claims)
            print(f"took {cc} as {a.id}")
            return
        if held and held["id"] != a.id:
            raise SystemExit(f"{cc} is held by {held['id']}, not {a.id}")
        if a.cmd == "park":
            HANDOFF.mkdir(exist_ok=True)
            h = HANDOFF / f"{cc}.md"
            if not h.exists():
                h.write_text(
                    f"# {cc} — parked {time.strftime('%Y-%m-%d %H:%M')} by {a.id}\n\n"
                    "Last checklist step done (AGENT_BRIEF.md §5):\n\n"
                    "What is on disk:\n\n"
                    "What I was about to do:\n\n"
                    "The one thing that will bite the next person:\n", encoding="utf-8")
            claims.pop(cc, None)
            write_claims(claims)
            set_status(cc, "parked")
            print(f"parked {cc}. FILL IN {h} before you stop.")
            return
        if a.cmd == "drop":
            claims.pop(cc, None)
            write_claims(claims)
            print(f"released {cc}")
            return
        # done
        sys.path.insert(0, str(ROOT))
        from countries import load_one
        try:
            load_one(cc)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"countries/{cc}.py does not load: {e}")
        if not (PROC / f"dots_{cc}.geojson").exists():
            raise SystemExit(f"no data/processed/dots_{cc}.geojson: run scatter.py --country {cc}")
        claims.pop(cc, None)
        write_claims(claims)
        set_status(cc, "drawn")
        (HANDOFF / f"{cc}.md").unlink(missing_ok=True)
        print(f"{cc} done. Still yours: sources/{cc}.md written, the build tail "
              "(tools/build_tail.py, unless a supervisor runs it), and your final report.")


if __name__ == "__main__":
    main()
