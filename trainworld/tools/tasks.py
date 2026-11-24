"""Task bookkeeping by command (T-3451 rec 3). Run from the maps root:
  venv\Scripts\python.exe trainworld/tools/tasks.py start <ids...> --agent <name> [--model m] [--tree t] [--what "..."]
  venv\Scripts\python.exe trainworld/tools/tasks.py done <id> --outcome "..." [--notes notes/T-x.md]
  venv\Scripts\python.exe trainworld/tools/tasks.py close <id> --reason "..." [--notes notes/T-x.md]
  venv\Scripts\python.exe trainworld/tools/tasks.py check
upnext add <ids> [--at N] / remove <ids> / list: the board's "Up next" queue (board-live.json "upnext"); start, done and close drop
their ids from it.
set --orchestrator "<id> (<name>)" / --build 8800=<hash>: the board header (the takeover and snap8800.py call it).
start: adds "(taking, <date>, agent <name>)" to each head line and the agent to notes/board-live.json.
done / close: moves the task block to the top of ARCHIVE.md, sets the line's done: in LINES.md when it was the
line's last open task, drops the agent from board-live.json when it has no other taken task, and marks the task's
agent reports read (tools/reports.py).
check: badly formed task lines, tags naming a line not in LINES.md, taking marks with no agent on the board, duplicate
ids, board agents whose task is not marked, a long handoff.md, a DECISIONS.md entry over 50 words. Exit 1 when it finds
anything.
--root <dir> runs on another copy of trainworld/ (for testing)."""
import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from filelock import FileLock

HEAD = re.compile(r"^- (T-\d+) \[(P\d)\] (.*)$")
TAKING = re.compile(r" \(taking, [^,)]*, agent ([^)]+)\)")
LTAG = re.compile(r"\{L:([^}]*)\}")
ARCHIVE_ANCHOR = "New entries go at the top, under this line."
DECISION_WORDS = 50  # a DECISIONS.md entry, counted after its "- " (T-4120)
FIELD_WORDS = {"Need from Anita": 30, "Context": 60, "Goal": 30}   # CLAUDE.md Tasks rules


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        s = f.read()
    eol = "\r\n" if "\r\n" in s else "\n"
    return s.replace("\r\n", "\n"), eol


def write(path, text, eol):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", eol))
    # Windows refuses the replace while another process has the file open for a moment (a reader, an editor)
    for wait in (0.2, 0.5, 1, 2, 0):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if not wait:
                raise
            time.sleep(wait)


def block_end(lines, i):
    """Index after the head line at i and its indented field lines."""
    j = i + 1
    while j < len(lines) and lines[j].startswith(" "):
        j += 1
    return j


def find_head(lines, tid):
    hits = [i for i, l in enumerate(lines) if l.startswith(f"- {tid} [")]
    return hits


class Root:
    def __init__(self, root):
        self.root = root
        self.tasks = os.path.join(root, "TASKS.md")
        self.archive = os.path.join(root, "ARCHIVE.md")
        self.lines = os.path.join(root, "LINES.md")
        self.live = os.path.join(root, "notes", "board-live.json")
        self.handoff = os.path.join(root, "notes", "handoff.md")
        self.decisions = os.path.join(root, "DECISIONS.md")
        self.lock = os.path.join(root, "notes", "tasks.lock")

    def locked(self):
        return FileLock(self.lock, err=SystemExit, message=f"{self.lock} is held; if no tasks.py is running, delete it")

    def board(self):
        with open(self.live, encoding="utf-8") as f:
            return json.load(f)

    def save_board(self, b):
        b["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        tmp = self.live + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(b, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, self.live)

    def taken_by(self, text):
        """agent name -> task ids marked taking in TASKS text."""
        out = {}
        for l in text.split("\n"):
            m = HEAD.match(l)
            if m:
                for a in TAKING.findall(m.group(3)):
                    out.setdefault(a.strip(), []).append(m.group(1))
        return out


def drop_upnext(b, ids):
    """Remove ids from the queue in board dict b; returns the ids that were there."""
    q = b.get("upnext") or []
    gone = [t for t in q if t in ids]
    if gone:
        b["upnext"] = [t for t in q if t not in ids]
    return gone


def open_ids(r):
    text, _ = read(r.tasks)
    return {m.group(1) for m in (HEAD.match(l) for l in text.split("\n")) if m}


def cmd_upnext(r, a):
    b = r.board()
    q = list(b.get("upnext") or [])
    if a.action == "list":
        print("\n".join(f"{i}. {t}" for i, t in enumerate(q, 1)) or "queue is empty")
        return 0
    ids = []
    for t in a.ids:
        t = t.strip()
        if t and t not in ids:
            ids.append(t)
    if a.action == "add":
        have = open_ids(r)
        new = []
        for t in ids:
            if t not in have:
                print(f"{t}: not an open task, skipped")
            elif t in q:
                print(f"{t}: already queued, skipped")
            else:
                new.append(t)
        if not new:
            return 1
        at = len(q) if a.at is None else max(0, min(len(q), a.at - 1))
        q[at:at] = new
        print(f"queued {', '.join(new)} at {at + 1}")
    else:
        gone = [t for t in ids if t in q]
        for t in ids:
            if t not in q:
                print(f"{t}: not queued")
        if not gone:
            return 1
        q = [t for t in q if t not in gone]
        print(f"removed {', '.join(gone)}")
    b["upnext"] = q
    r.save_board(b)
    return 0


def cmd_start(r, a):
    text, eol = read(r.tasks)
    lines = text.split("\n")
    today = date.today().isoformat()
    marked = []
    for tid in a.ids:
        hits = find_head(lines, tid)
        if len(hits) != 1:
            print(f"{tid}: {'not found' if not hits else 'several head lines'}, skipped")
            continue
        l = lines[hits[0]]
        if TAKING.search(l):
            print(f"{tid}: already marked taking, skipped")
            continue
        if l.endswith(" Open."):
            l = l[: -len(" Open.")]
        lines[hits[0]] = f"{l} (taking, {today}, agent {a.agent})"
        marked.append(tid)
        print(f"{tid}: marked")
    if not marked:
        return 1
    write(r.tasks, "\n".join(lines), eol)
    b = r.board()
    mine = next((x for x in b["agents"] if x["name"] == a.agent), None)
    if mine is None:
        mine = {"name": a.agent, "task": "", "model": a.model or "", "what": a.what or "",
                "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}   # board.js shows it as "N min"
        b["agents"].append(mine)
    have = [t.strip() for t in mine["task"].split(",") if t.strip()]
    mine["task"] = ", ".join(have + [t for t in marked if t not in have])
    if a.model:
        mine["model"] = a.model
    if a.what:
        mine["what"] = a.what
    if a.tree:
        mine["tree"] = a.tree
    gone = drop_upnext(b, marked)
    r.save_board(b)
    if gone:
        print(f"up next: dropped {', '.join(gone)}")
    print(f"board: {a.agent} on {mine['task']}")
    return 0


def cmd_set(r, a):
    """Board header fields the orchestrator and the 8800 rebuild keep current."""
    b = r.board()
    if a.orchestrator:
        b["orchestrator"] = a.orchestrator
    if a.build:
        k, _, v = a.build.partition("=")
        b.setdefault("builds", {})[k] = v
    r.save_board(b)
    print(f"board: orchestrator {b.get('orchestrator')}; builds {b.get('builds')}")
    return 0


def finish(r, a, how):
    text, eol = read(r.tasks)
    lines = text.split("\n")
    hits = find_head(lines, a.id)
    if len(hits) != 1:
        print(f"{a.id}: {'not found in TASKS.md' if not hits else 'several head lines'}")
        return 1
    i = hits[0]
    j = block_end(lines, i)
    head = lines[i]
    m = HEAD.match(head)
    pri, rest = m.group(2), m.group(3)
    agents = [x.strip() for x in TAKING.findall(rest)]
    rest = TAKING.sub("", rest)
    rest = re.sub(r"\{(waits|after):[^}]*\}\s*", "", rest)
    slug = (LTAG.search(rest) or [None, None])[1]
    # a head line carrying its whole old-shape text keeps only what precedes the first sentence end
    today = date.today().isoformat()
    body = (a.outcome if how == "done" else f"closed without code: {a.reason}").strip()
    if not body.endswith((".", ")")):
        body += "."
    entry = f"- {a.id} [{pri}] {rest.strip()} ({today}): {body}"
    if a.notes:
        entry += f" {a.notes}."
    # archive first: a failure between the two writes leaves the task in both files, never in neither
    atext, aeol = read(r.archive)
    k = atext.find(ARCHIVE_ANCHOR)
    if k < 0:
        print("ARCHIVE.md: anchor line not found; nothing changed. Entry:\n" + entry)
        return 1
    k = atext.index("\n", k) + 1
    write(r.archive, atext[:k] + entry + "\n" + atext[k:], aeol)
    del lines[i:j]
    write(r.tasks, "\n".join(lines), eol)
    print(f"{a.id}: archived")

    if slug:
        still = [l for l in lines if HEAD.match(l) and LTAG.search(l) and LTAG.search(l).group(1) == slug]
        if not still:
            ltext, leol = read(r.lines)
            ll = ltext.split("\n")
            for n, l in enumerate(ll):
                if l.startswith(f"- {slug} |"):
                    if "| done:" in l:
                        print(f"line {slug}: already done")
                    else:
                        ll[n] = f"{l} | done:{today}"
                        write(r.lines, "\n".join(ll), leol)
                        print(f"line {slug}: done:{today} (it was the last open task)")
                    break
            else:
                print(f"line {slug}: not in LINES.md")
    else:
        print("no {L:} tag on the task; LINES.md untouched")

    taken = r.taken_by("\n".join(lines))
    b = r.board()
    for ag in agents:
        mine = next((x for x in b["agents"] if x["name"] == ag), None)
        if mine is None:
            continue
        left = taken.get(ag, [])
        if left:
            mine["task"] = ", ".join(left)
            print(f"board: {ag} still on {mine['task']}")
        else:
            b["agents"].remove(mine)
            print(f"board: {ag} dropped")
    gone = drop_upnext(b, {a.id})
    if gone:
        print("up next: dropped " + a.id)
    if agents or gone:
        r.save_board(b)
    sys.path.insert(0, HERE)
    import reports
    n = reports.Box(r.root).read_task(a.id, f"tasks.py {how}")
    if n:
        print(f"agent reports: {n} for {a.id} marked read")
    print(f"Still to do by hand: grep TASKS.md for {a.id}; close a parent whose last stage this was; "
          "name other open tasks it also did; DECISIONS.md if a choice was made.")
    return 0


def cmd_check(r, a):
    problems = []
    text, _ = read(r.tasks)
    lines = text.split("\n")
    ltext, _ = read(r.lines)
    slugs = {l[2:].split(" |")[0].strip() for l in ltext.split("\n") if l.startswith("- ") and " | " in l}
    try:
        agents = {x["name"]: x for x in r.board()["agents"]}
    except Exception as e:
        agents, _e = {}, problems.append(f"board-live.json: {e}")
    seen = {}
    taken = {}
    section = ""
    for n, l in enumerate(lines, 1):
        if l.startswith("#"):
            section = l.strip("# ")
        m = HEAD.match(l)
        if not m:
            if l.startswith("- T-"):
                problems.append(f"TASKS.md:{n}: head line not 'T-id [P#] ...': {l[:70]}")
            continue
        tid, pri, rest = m.groups()
        where = f"TASKS.md:{n} {tid}"
        if tid in seen:
            problems.append(f"{where}: duplicate id (also line {seen[tid]})")
        seen[tid] = n
        for ag in (x.strip() for x in TAKING.findall(rest)):
            taken.setdefault(ag, []).append(tid)
            if ag not in agents:
                problems.append(f"{where}: taking mark for agent '{ag}' who is not in board-live.json")
        title = TAKING.sub("", rest)
        lm = LTAG.search(title)
        if not lm:
            problems.append(f"{where}: no {{L:slug}} tag")
        elif lm.group(1) not in slugs:
            problems.append(f"{where}: line '{lm.group(1)}' is not in LINES.md")
        elif not title.startswith(lm.group(0)):
            problems.append(f"{where}: {{L:}} tag is not right after the priority")
        tags = re.findall(r"\{(?:L|waits|after):[^}]*\}", title)
        plain = title
        for tg in tags:
            plain = plain.replace(tg, "")
        plain = plain.strip()
        words = len(plain.split())
        if words > 14 or len(plain) > 110:
            problems.append(f"{where}: old shape or long head line ({words} words); needs head plus Context field")
        elif words < 2:
            problems.append(f"{where}: title under 2 words")
        elif plain.endswith((".", ":")) or ": " in plain:
            problems.append(f"{where}: title has a full stop or colon")
        j = block_end(lines, n - 1)
        fields = [re.match(r"^  - ([A-Za-z ]+):", x) for x in lines[n:j]]
        names = [f.group(1) for f in fields if f]
        if len(plain.split()) <= 14:
            order = ["Need from Anita", "Context", "Goal", "Other"]
            # Goal is optional when the title already says what done is (Anita, 2026-11-18)
            if "Context" not in names:
                problems.append(f"{where}: missing Context field")
            for x in lines[n:j]:
                fm = re.match(r"^  - (Need from Anita|Context|Goal):(.*)$", x)
                if fm and len(fm.group(2).split()) > FIELD_WORDS[fm.group(1)]:
                    problems.append(f"{where}: {fm.group(1)} is {len(fm.group(2).split())} words (at most {FIELD_WORDS[fm.group(1)]})")
            bad = [x for x in names if x not in order]
            if bad:
                problems.append(f"{where}: unknown field {bad[0]}")
            elif names != sorted(names, key=order.index):
                problems.append(f"{where}: fields out of order {names}")
            if "{waits:Anita}" in title and "Need from Anita" not in names:
                problems.append(f"{where}: {{waits:Anita}} without a Need from Anita field")
            if "Need from Anita" in names and "{waits:Anita}" not in title:
                problems.append(f"{where}: Need from Anita field without {{waits:Anita}}")
    atext, _ = read(r.archive)
    for tid in seen:
        if re.search(rf"^- {tid} \[", atext, re.M):
            problems.append(f"{tid}: open in TASKS.md and archived in ARCHIVE.md")
    for name, ag in agents.items():
        if not taken.get(name):
            problems.append(f"board-live.json: agent '{name}' ({ag.get('task')}) has no task marked taking")
    if os.path.exists(r.handoff):
        n = len(read(r.handoff)[0].split("\n"))
        if n > 60:
            problems.append(f"notes/handoff.md is {n} lines (keep it under 60)")
    if os.path.exists(r.decisions):
        for n, l in enumerate(read(r.decisions)[0].split("\n"), 1):
            words = len(l[2:].split())
            if l.startswith("- ") and words > DECISION_WORDS:
                problems.append(f"DECISIONS.md:{n}: {words} words (an entry is one or two sentences, at most {DECISION_WORDS})")
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s), {len(seen)} open tasks")
    return 1 if problems else 0


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(HERE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("ids", nargs="+")
    s.add_argument("--agent", required=True)
    s.add_argument("--model")
    s.add_argument("--tree")
    s.add_argument("--what")
    d = sub.add_parser("done")
    d.add_argument("id")
    d.add_argument("--outcome", required=True)
    d.add_argument("--notes")
    c = sub.add_parser("close")
    c.add_argument("id")
    c.add_argument("--reason", required=True)
    c.add_argument("--notes")
    u = sub.add_parser("upnext")
    u.add_argument("action", choices=["add", "remove", "list"])
    u.add_argument("ids", nargs="*")
    u.add_argument("--at", type=int, help="1-based place in the queue (default: the end)")
    t = sub.add_parser("set")
    t.add_argument("--orchestrator", help='"<session id short> (<name>)", shown on the board')
    t.add_argument("--build", help='"8800=<hash>" or "master=<hash>"')
    sub.add_parser("check")
    a = ap.parse_args()
    r = Root(a.root)
    cmds = {"start": lambda: cmd_start(r, a), "done": lambda: finish(r, a, "done"),
            "close": lambda: finish(r, a, "close"), "check": lambda: cmd_check(r, a), "upnext": lambda: cmd_upnext(r, a),
            "set": lambda: cmd_set(r, a)}
    writes = a.cmd in ("start", "done", "close", "set") or (a.cmd == "upnext" and a.action != "list")
    if writes:
        with r.locked():
            code = cmds[a.cmd]()
    else:
        code = cmds[a.cmd]()
    sys.exit(code)


main()
