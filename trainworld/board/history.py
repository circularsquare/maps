"""Open-task history for the board's graph (T-3504).

  python trainworld/board/history.py record [--force]   append one row if the last is 50+ minutes old (serve.py runs
                                                         this hourly); also rewrites board/history.json
  python trainworld/board/history.py backfill [--days 7] [--redo] rebuild the rows older than the first recorded one, from
                                                         TASKS.md and LINES.md at each hour in git history
  python trainworld/board/history.py show                print the last rows

notes/task-history.jsonl: one row per line {t, n, area{slug: n}, pri{P1..P4, none: n}, opened, closed}. "Open" is what
the board counts: every task in TASKS.md except a parked one (P4 or under Someday, nobody on it, not waiting). opened/closed are the
task ids that appeared in TASKS.md / vanished from it into ARCHIVE.md since the previous row. A task that
vanishes without an ARCHIVE.md entry (dropped, merged, moved elsewhere) is not a close. ARCHIVE.md is add-only, so
the backfill checks vanished ids against today's file.
Areas come from the task's line in LINES.md; before LINES.md existed (2026-11-15) a task takes the area it has in the
earliest snapshot that has one, else "none" (Not sorted).
notes/task-history-ids.json is the id list of the last row; board/history.json is what the page reads.
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402

TW = HERE.parent
REPO = TW.parent
ROWS = TW / "notes" / "task-history.jsonl"
IDS = TW / "notes" / "task-history-ids.json"
OUT = HERE / "history.json"
KEEP_DAYS = 14
MIN_GAP = 50 * 60


def snapshot(tasks_text, lines_text, fallback=None):
    """-> (counts row without t, ids in TASKS.md, {id: area} for tasks with a line)."""
    tasks = build.parse_tasks(tasks_text or "")
    slug_area = {l["slug"]: l["area"] for l in build.parse_lines(lines_text)[0]} if lines_text else {}
    area, pri, ids, seen, own = {}, {}, set(), set(), {}
    for t in tasks:
        if t["id"] in seen:
            continue
        seen.add(t["id"])
        ids.add(t["id"])
        st = build.task_status(t, {})
        a = slug_area.get(t["lines"][0]) if t["lines"] else None
        if lines_text is not None and a is None:
            a = "ops"                         # the board puts a task with no line in Not sorted, in the ops row
        if a:
            own[t["id"]] = a
        if (t["pri"] == "P4" or t["someday"]) and st in ("ready", "after"):   # Someday was P3 before 2026-11-16
            continue
        a = a or (fallback or {}).get(t["id"]) or "none"
        area[a] = area.get(a, 0) + 1
        p = t["pri"] or "none"
        pri[p] = pri.get(p, 0) + 1
    return {"n": sum(area.values()), "area": area, "pri": pri}, ids, own


def archived_ids():
    text = build.read(TW / "ARCHIVE.md") or ""
    return set(re.findall(r"^- (T-\d+) ", text, re.M))


def read_rows():
    if not ROWS.exists():
        return []
    return [json.loads(l) for l in ROWS.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_rows(rows):
    tmp = ROWS.with_name(ROWS.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    os.replace(tmp, ROWS)


def write_page(rows):
    cut = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=KEEP_DAYS)).strftime("%Y-%m-%dT%H:%MZ")
    data = {"areas": [{"slug": s, "name": n} for s, n in build.AREAS] + [{"slug": "none", "name": "Not sorted"}],
            "rows": [r for r in rows if r["t"] >= cut]}
    tmp = OUT.with_name(OUT.name + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, OUT)


def now_stamp():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def record(force=False):
    rows = read_rows()
    if rows and not force:
        last = dt.datetime.strptime(rows[-1]["t"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc)
        if (dt.datetime.now(dt.timezone.utc) - last).total_seconds() < MIN_GAP:
            write_page(rows)
            return None
    tasks_text = build.read(TW / "TASKS.md")
    if tasks_text is None:
        sys.exit("no TASKS.md")
    lines_text = build.read(TW / "LINES.md")
    row, ids, own = snapshot(tasks_text, lines_text)
    row = {"t": now_stamp(), **row}
    try:
        before = set(json.loads(IDS.read_text(encoding="utf-8")))
        row["opened"] = len(ids - before)
        row["closed"] = len((before - ids) & archived_ids())
    except (OSError, ValueError):
        pass
    rows.append(row)
    write_rows(rows)
    IDS.write_text(json.dumps(sorted(ids)), encoding="utf-8")
    write_page(rows)
    return row


def git(*args):
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, encoding="utf-8")
    return r.stdout


def backfill(days=7):
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    existing = read_rows()
    stop = existing[0]["t"] if existing else None
    if existing and "--redo" in sys.argv:
        existing = []
        stop = None
    points = [now - dt.timedelta(hours=h) for h in range(days * 24, -1, -1)]
    points = [p for p in points if not stop or p.strftime("%Y-%m-%dT%H:%MZ") < stop]
    blobs = {}

    def show(commit, path):
        k = (commit, path)
        if k not in blobs:
            blobs[k] = git("show", f"{commit}:{path}")
        return blobs[k]

    done = archived_ids()
    snaps = []
    for p in points:
        iso = p.strftime("%Y-%m-%dT%H:%M:%SZ")
        tc = git("log", "-1", "--format=%H", f"--before={iso}", "HEAD", "--", "trainworld/TASKS.md").strip()
        lc = git("log", "-1", "--format=%H", f"--before={iso}", "HEAD", "--", "trainworld/LINES.md").strip()
        if not tc:
            continue
        snaps.append((p, show(tc, "trainworld/TASKS.md"), show(lc, "trainworld/LINES.md") if lc else None))
    # for the hours before LINES.md existed: the area a task has in the earliest snapshot that names one
    fallback = {}
    for p, tt, lt in snaps:
        if lt:
            for k, v in snapshot(tt, lt)[2].items():
                fallback.setdefault(k, v)
    rows, prev = [], None
    for p, tt, lt in snaps:
        row, ids, _ = snapshot(tt, lt, fallback)
        row = {"t": p.strftime("%Y-%m-%dT%H:%MZ"), **row}
        if prev is not None:
            row["opened"] = len(ids - prev)
            row["closed"] = len((prev - ids) & done)
        prev = ids
        rows.append(row)
    write_rows(rows + existing)
    if not existing and prev is not None:
        IDS.write_text(json.dumps(sorted(prev)), encoding="utf-8")
    write_page(rows + existing)
    return len(rows)


def main():
    a = sys.argv[1:]
    if a[:1] == ["record"]:
        r = record("--force" in a)
        print(json.dumps(r) if r else "last row is recent; no row added")
    elif a[:1] == ["backfill"]:
        days = int(a[a.index("--days") + 1]) if "--days" in a else 7
        print(f"{backfill(days)} rows")
    elif a[:1] == ["show"]:
        for r in read_rows()[-10:]:
            print(r)
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
