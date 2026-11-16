"""Task bookkeeping by command (T-3451 rec 3). Run from the maps root:
  venv\Scripts\python.exe trainworld/tools/tasks.py start <ids...> --agent <name> [--model m] [--tree t] [--what "..."]
  venv\Scripts\python.exe trainworld/tools/tasks.py done <id> --outcome "..." [--notes notes/T-x.md]
  venv\Scripts\python.exe trainworld/tools/tasks.py close <id> --reason "..." [--notes notes/T-x.md]
  venv\Scripts\python.exe trainworld/tools/tasks.py check
upnext add <ids> [--at N] / remove <ids> / list: the board's "Up next" queue (board-live.json "upnext"); start, done and close drop
their ids from it.
start: adds "(taking, <date>, agent <name>)" to each head line and the agent to notes/board-live.json.
done / close: moves the task block to the top of ARCHIVE.md, sets the line's done: in LINES.md when it was the
line's last open task, and drops the agent from board-live.json when it has no other taken task.
check: badly formed task lines, tags naming a line not in LINES.md, taking marks with no agent on the board, duplicate
ids, board agents whose task is not marked, a long handoff.md. Exit 1 when it finds anything.
--root <dir> runs on another copy of trainworld/ (for testing)."""
import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
HEAD = re.compile(r"^- (T-\d+) \[(P\d)\] (.*)$")
TAKING = re.compile(r" \(taking, [^,)]*, agent ([^)]+)\)")
LTAG = re.compile(r"\{L:([^}]*)\}")
ARCHIVE_ANCHOR = "New entries go at the top, under this line."


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        s = f.read()
    eol = "\r\n" if "\r\n" in s else "\n"
    return s.replace("\r\n", "\n"), eol


def write(path, text, eol):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", eol))
    os.replace(tmp, path)


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
        mine = {"name": a.agent, "task": "", "model": a.model or "", "what": a.what or ""}
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
    del lines[i:j]
    write(r.tasks, "\n".join(lines), eol)

    atext, aeol = read(r.archive)
    k = atext.find(ARCHIVE_ANCHOR)
    if k < 0:
        print("ARCHIVE.md: anchor line not found; entry NOT archived, task already removed. Entry:\n" + entry)
        return 1
    k = atext.index("\n", k) + 1
    write(r.archive, atext[:k] + entry + "\n" + atext[k:], aeol)
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
            problems.append(f"{where}: old shape or long head line ({words} words); needs head plus Context/Goal fields")
        elif words < 2:
            problems.append(f"{where}: title under 2 words")
        elif plain.endswith((".", ":")) or ": " in plain:
            problems.append(f"{where}: title has a full stop or colon")
        j = block_end(lines, n - 1)
        fields = [re.match(r"^  - ([A-Za-z ]+):", x) for x in lines[n:j]]
        names = [f.group(1) for f in fields if f]
        if len(plain.split()) <= 14:
            order = ["Need from Anita", "Context", "Goal", "Other"]
            if "Context" not in names or "Goal" not in names:
                problems.append(f"{where}: missing Context or Goal field")
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
        if n > 80:
            problems.append(f"notes/handoff.md is {n} lines (keep it under 80)")
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
    sub.add_parser("check")
    a = ap.parse_args()
    r = Root(a.root)
    code = {"start": lambda: cmd_start(r, a), "done": lambda: finish(r, a, "done"),
            "close": lambda: finish(r, a, "close"), "check": lambda: cmd_check(r, a), "upnext": lambda: cmd_upnext(r, a)}[a.cmd]()
    sys.exit(code)


main()
