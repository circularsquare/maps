"""Writes trainworld/board/data.json for the board page (twboard on 8800).

Reads TASKS.md, LINES.md, notes/board-live.json, the core ledger and git HEAD. The format and the
status rules are in notes/T-3389.md "Format (decided)"; notes/T-3391.md says what is where.

    venv\\Scripts\\python.exe trainworld/board/build.py [--tasks F] [--lines F] [--live F] [--out F]

The flags exist for testing against fixtures; serve.py runs it with none, cwd = the maps root.
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TW = HERE.parent
MAPS = TW.parent

AREAS = [
    ("track", "Track and stations"),
    ("schedule", "Scheduling and fares"),
    ("demand", "Demand and world data"),
    ("schematic", "Schematic"),
    ("community", "Community and accounts"),
    ("interface", "Interface"),
    ("performance", "Performance"),
    ("ops", "Ops and code health"),
]
# Board order of columns; a line takes the first of these that any of its open tasks has, except
# that "done" is only for lines with no open tasks.
COLUMNS = [
    ("progress", "In progress"),
    ("you", "Waiting on you"),
    ("high", "Ready (high)"),
    ("low", "Ready (low)"),
    ("after", "Waiting on another task"),
    ("done", "Done this week"),
]
LINE_ORDER = ["progress", "you", "high", "low", "after"]
UPDATES = {"track-stations": "Track and stations update", "scheduling-fares": "Scheduling and fares update"}
UNSORTED = "not-sorted"
DONE_DAYS = 7

TASK_RE = re.compile(r"^- (T-\d+)\b\s*(.*)$")
PRI_RE = re.compile(r"^\[(P\d)\]\s*")
TAG_RE = re.compile(r"\{\s*(L|waits|after)\s*:\s*([^}]*)\}\s*")
HEAD_RE = re.compile(r"^(#{2,6})\s+(.*?)\s*$")
AREA_HEAD_RE = re.compile(r"\{area:([a-z-]+)\}")
TAKING_RE = re.compile(r"\btaking,")      # the mark is "taking, <date>"; the bare word turns up in quotes


def read(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError:
        return None


def parse_tasks(text):
    """Every task line: id, priority, tags, the text without tags, its section, Someday or not."""
    tasks, heads = [], {}
    for raw in text.splitlines():
        h = HEAD_RE.match(raw)
        if h:
            level = len(h.group(1))
            heads = {k: v for k, v in heads.items() if k < level}
            heads[level] = h.group(2)
            continue
        m = TASK_RE.match(raw)
        if not m:
            continue
        tid, rest = m.groups()
        pri = None
        p = PRI_RE.match(rest)
        if p:
            pri, rest = p.group(1), rest[p.end():]
        lines, waits, after = [], [], []
        # tags sit at the front, but take them from anywhere so a misplaced one still counts
        for kind, val in TAG_RE.findall(rest):
            val = val.strip()
            if kind == "L":
                lines.append(val)
            elif kind == "waits":
                waits.append(val)
            else:
                after.append(val)
        body = TAG_RE.sub("", rest).strip()
        if pri is None:                      # a priority written after the tags
            p = PRI_RE.match(body)
            if p:
                pri, body = p.group(1), body[p.end():]
        section = " / ".join(heads[k] for k in sorted(heads))
        tasks.append({
            "id": tid, "pri": pri, "text": body, "section": section,
            "someday": heads.get(2, "").strip().lower() == "someday",
            "lines": lines, "waits": waits, "after": after,
            "taking": bool(TAKING_RE.search(body)),
        })
    return tasks


def parse_lines(text):
    """LINES.md: one entry per line, under the heading of its area."""
    out, area, warnings = [], None, []
    for raw in (text or "").splitlines():
        h = HEAD_RE.match(raw)
        if h:
            a = AREA_HEAD_RE.search(h.group(2))
            area = a.group(1) if a else None
            continue
        if not raw.startswith("- ") or "|" not in raw:
            continue
        parts = [p.strip() for p in raw[2:].split("|")]
        if len(parts) < 3:
            warnings.append(f"LINES.md: too few fields: {raw[:60]}")
            continue
        entry = {"slug": parts[0], "name": parts[1], "about": parts[2], "area": area,
                 "update": None, "done": None}
        for extra in parts[3:]:
            k, _, v = extra.partition(":")
            k, v = k.strip(), v.strip()
            if k == "update":
                entry["update"] = v
            elif k == "done":
                entry["done"] = v
            elif extra:
                warnings.append(f"LINES.md {parts[0]}: unknown field {extra}")
        if area not in dict(AREAS):
            warnings.append(f"LINES.md {parts[0]}: not under a known area heading")
            entry["area"] = "ops"
        out.append(entry)
    return out, warnings


def task_status(t, busy):
    if t["id"] in busy or t["taking"]:
        return "progress"
    if t["waits"]:
        return "you"
    if t["after"]:
        return "after"
    if t["pri"] in ("P1", "P2") and not t["someday"]:
        return "high"
    return "low"


def cores():
    """The ledger as `cores.py list` prints it: claims, claimed, cap, machine."""
    try:
        out = subprocess.run([sys.executable, str(MAPS / "cores.py"), "list"], cwd=MAPS,
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    res = {"claims": [], "claimed": None, "cap": None, "machine": None}
    for ln in out.splitlines():
        s = re.match(r"claimed (\d+) of cap (\d+) \(machine (\d+)\)", ln)
        if s:
            res["claimed"], res["cap"], res["machine"] = map(int, s.groups())
            continue
        c = re.match(r"(\S+)\s+(\d+) cores?\s+(?:pid (\d+)|no pid)\s+(\S+)\s*(.*)$", ln)
        if c:
            res["claims"].append({"name": c.group(1), "cores": int(c.group(2)),
                                  "pid": int(c.group(3)) if c.group(3) else None,
                                  "age": c.group(4), "note": c.group(5).strip()})
    return res


def git_head():
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=MAPS,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(TW / "TASKS.md"))
    ap.add_argument("--lines", default=str(TW / "LINES.md"))
    ap.add_argument("--live", default=str(TW / "notes" / "board-live.json"))
    ap.add_argument("--out", default=str(HERE / "data.json"))
    args = ap.parse_args()
    now = dt.datetime.now(dt.timezone.utc)

    tasks_text = read(args.tasks)
    if tasks_text is None:
        sys.exit(f"no {args.tasks}")
    tasks = parse_tasks(tasks_text)
    lines_text = read(args.lines)
    lines, warnings = parse_lines(lines_text)
    if lines_text is None:
        warnings.append("LINES.md is missing: every task is in Not sorted yet")

    try:
        live = json.loads(read(args.live) or "{}")
    except ValueError:
        live = {}
        warnings.append("board-live.json does not parse")
    agents = live.get("agents") or []
    busy = {}
    for a in agents:
        for tid in re.findall(r"T-\d+", str(a.get("task", ""))):
            busy.setdefault(tid, a.get("name"))

    by_slug = {}
    for ln in lines:
        if ln["slug"] in by_slug:
            warnings.append(f"LINES.md: {ln['slug']} twice")
            continue
        ln["tasks"] = []
        by_slug[ln["slug"]] = ln
    unsorted = {"slug": UNSORTED, "name": "Not sorted yet", "about": "Tasks with no line yet.",
                "area": "ops", "update": None, "done": None, "tasks": []}

    notes_index = index_notes()
    seen, task_out = set(), {}
    for t in tasks:
        if t["id"] in seen:
            warnings.append(f"{t['id']} is in TASKS.md twice")
            continue
        seen.add(t["id"])
        t["status"] = task_status(t, busy)
        t["agent"] = busy.get(t["id"])
        slug = t["lines"][0] if t["lines"] else None
        if len(t["lines"]) > 1:
            warnings.append(f"{t['id']} has {len(t['lines'])} line tags")
        if slug and slug not in by_slug:
            warnings.append(f"{t['id']} names a line that LINES.md does not have: {slug}")
            slug = None
        (by_slug[slug] if slug else unsorted)["tasks"].append(t["id"])
        t["line"] = slug or UNSORTED
        del t["lines"], t["taking"]
        t["notes"] = task_notes(t["id"], t["text"], notes_index)
        task_out[t["id"]] = t

    out_lines = []
    for ln in list(by_slug.values()) + ([unsorted] if unsorted["tasks"] else []):
        counts = {k: 0 for k, _ in COLUMNS}
        for tid in ln["tasks"]:
            counts[task_out[tid]["status"]] += 1
        ln["counts"] = counts
        ln["open"] = len(ln["tasks"])
        status = next((k for k in LINE_ORDER if counts[k]), None)
        if status is None:
            status = "done" if recent(ln["done"], now) else None
        ln["status"] = status          # None: an old done line, or one with no tasks; not shown
        out_lines.append(ln)

    data = {
        "built": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head": git_head(),
        "areas": [{"slug": s, "name": n} for s, n in AREAS],
        "columns": [{"key": k, "name": n} for k, n in COLUMNS],
        "updates": [{"key": k, "name": n} for k, n in UPDATES.items()],
        "lines": out_lines,
        "tasks": task_out,
        "live": {"updated": live.get("updated"), "orchestrator": live.get("orchestrator"),
                 "builds": live.get("builds") or {}, "agents": agents, "runs": live.get("runs") or []},
        "cores": cores(),
        "warnings": warnings,
    }
    archive = archive_entries(notes_index)
    data["archived"] = sorted(int(k[2:]) for k in archive if k not in task_out)
    data["notes_index"] = notes_index
    req = requests_data(now, warnings)
    write_json(args.out, data)
    write_json(str(Path(args.out).with_name("requests.json")), req)
    write_json(str(Path(args.out).with_name("archive.json")), archive)
    shown = sum(1 for ln in out_lines if ln["status"])
    print(f"{len(task_out)} tasks, {len(out_lines)} lines ({shown} shown), "
          f"{len(unsorted['tasks'])} not sorted, {len(req['groups'])} request groups, {len(warnings)} warnings")


NOTES = TW / "notes"
NOTES_MENTION = re.compile(r"notes/(?:archive/[\w-]+/)?([\w.-]+?\.md|T-\d+/)")


def index_notes():
    """Notes the board can show, by name as tasks cite them: "T-3390.md", "someday.md", or "T-1114/" (a folder:
    its .md files). Values are paths under notes/ (served as twnotes); a current note wins over an archived one."""
    index = {}
    places = [NOTES] + sorted(p for p in (NOTES / "archive").glob("*") if p.is_dir())
    for place in reversed(places):              # current last, so it overwrites
        for p in place.iterdir():
            rel = p.relative_to(NOTES).as_posix()
            if p.is_file() and p.suffix == ".md":
                index[p.name] = rel
            elif p.is_dir() and re.fullmatch(r"T-\d+", p.name):
                files = sorted(f.relative_to(NOTES).as_posix() for f in p.rglob("*.md"))
                if files:
                    index[p.name + "/"] = files
    return index


def task_notes(tid, text, index):
    """The task's notes: its own file if the text cites it (T-<id>.md, T-<id>-*.md or the T-<id>/ folder), else the
    first notes file it cites (another task's, someday.md), else T-<id>.md if one exists."""
    cited = [m for m in NOTES_MENTION.findall(text) if m in index]
    own = [m for m in cited if re.match(re.escape(tid) + r"(\.md|/|-)", m)]
    for name in own + cited + [f"{tid}.md", f"{tid}/"]:
        if name in index:
            return name
    return None


def archive_entries(index):
    """Closed tasks from ARCHIVE.md and ARCHIVE-to-2026-10-19.md, newest file first: {id: {text, file, notes}}."""
    out = {}
    for name in ("ARCHIVE.md", "ARCHIVE-to-2026-10-19.md"):
        for raw in (read(TW / name) or "").splitlines():
            m = TASK_RE.match(raw)
            if m and m.group(1) not in out:
                tid, text = m.group(1), raw[2:].strip()
                out[tid] = {"text": text, "file": name, "notes": task_notes(tid, text, index)}
    return out


def write_json(path, obj):
    tmp = Path(path + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)


def jsonl(path):
    out = []
    for line in (read(path) or "").splitlines():
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out


def requests_data(now, warnings):
    """Player requests (T-3393), kept out of data.json: the groups, the reports they hold, and pasted items
    by inbox id (a paste stays readable here after the orchestrator marks it handled)."""
    try:
        doc = json.loads(read(TW / "notes" / "board-requests.json") or "{}")
    except ValueError:
        doc = {}
        warnings.append("board-requests.json does not parse")
    reports = {r["id"]: r for r in jsonl(TW / "notes" / "board-reports.jsonl") if "id" in r}
    pastes = {r["id"]: r for r in jsonl(TW / "notes" / "board-inbox.jsonl")
              if r.get("type") == "paste" and "id" in r}
    groups = doc.get("groups") or []
    for g in groups:
        for rid in g.get("reports", []):
            if rid not in reports:
                warnings.append(f"request group {g.get('id')} names a report not fetched: {rid}")
    grouped = {rid for g in groups for rid in g.get("reports", [])}
    return {
        "built": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sorted_at": doc.get("sorted_at"),
        "groups": groups,
        "reports": reports,
        "pastes": pastes,
        "unsorted_reports": [rid for rid in reports if rid not in grouped],
    }


def recent(day, now):
    if not day:
        return False
    try:
        d = dt.date.fromisoformat(day)
    except ValueError:
        return False
    return (now.date() - d).days <= DONE_DAYS


if __name__ == "__main__":
    main()
