"""The orchestrator's side of the board inbox (trainworld/notes/board-inbox.jsonl).

    python trainworld/board/watch.py --session <id> # keeps running; one line per batch that is ready
    python trainworld/board/watch.py pending      # her pending actions, one JSON line each
    python trainworld/board/watch.py idle         # the hourly idle line once, or "busy"
    python trainworld/board/watch.py handled <id> [<id> ...] --by <orchestrator> [--note "..."]

A batch is ready when something is pending and she pressed Send now after the last batch was
reported, or a pad-send (the scratch pad's Send for filing) is new, or nothing new has come from
her for --idle seconds (180). The watch prints only then,
for the Monitor tool. The line format and the action types (reorder, area, priority, move-task, answer,
ask-answer, text-ack, add, comment, ...) are in notes/T-3391.md.

It also prints a "disk:" line when free space on the drive falls under 30 GB, again under 10 GB, and once it is
back (checked every 5 minutes; tools/ops/disk.py), chiming for Anita with a Windows notification as it falls.
It also prints a line for each unread agent report (tools/reports.py, T-3668) that another session's agent
sent: the current orchestrator's own agents already reach it directly. --session registers this session for
the reports hook (ids already printed are kept per session in notes/watch-told-<session>.json, so the 30-minute
restarts don't reprint them), and the watch stops once notes/handoff.md names another orchestrator.
"""
import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import time
from pathlib import Path

TW = Path(__file__).resolve().parents[1]
INBOX = TW / "notes" / "board-inbox.jsonl"
sys.path.insert(0, str(TW / "tools"))
sys.path.insert(0, str(TW / "tools" / "ops"))
import reports  # noqa: E402
import disk  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")   # her text has CJK; Windows consoles default to cp1252


def rows(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue                 # a line still being written
                if isinstance(obj, dict):
                    out.append(obj)
    except FileNotFoundError:
        pass
    return out


def when(stamp):
    try:
        return dt.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc).timestamp()
    except (TypeError, ValueError):
        return None


def state(path):
    """(pending actions, time of her last line, time of her last Send now)"""
    rs = rows(path)
    handled = {str(r["handled"]) for r in rs if "handled" in r}
    pending, last, sent = [], None, None
    for r in rs:
        if "handled" in r and r.get("by") != "anita":
            continue                         # the orchestrator's marks are not her activity
        t = when(r.get("at"))
        if t:
            last = max(last or t, t)
        if r.get("type") == "send-now":
            sent = t
        elif "id" in r and str(r["id"]) not in handled:
            pending.append(r)
    return pending, last, sent


def told_file(session):
    return TW / "notes" / f"watch-told-{(session or 'none')[:8]}.json"


def load_told(session):
    try:
        return set(json.loads(told_file(session).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


def agent_reports(box, session, told):
    """Lines for unread agent reports not yet printed; None once another session orchestrates."""
    cur = box.current()
    if session and cur and not (cur.startswith(session) or session.startswith(cur)):
        return None
    out = []
    for r in box.unread():
        if r["id"] in told or (session and str(r.get("session", "")).startswith(session[:8])):
            continue
        told.add(r["id"])
        try:
            told_file(session).write_text(json.dumps(sorted(told)), encoding="utf-8")
        except OSError:
            pass
        out.append(f"agent report {reports.line(r)} (reports.py show {r['id']})")
    return out


ROOT = TW.parent
HEAD = re.compile(r"^- (T-\d+) \[(P\d)\] (.*)$")
IDLE_EVERY = 3600
DISK_EVERY = 300
MIN_AGENTS, MAX_AGENTS, MIN_FREE = 2, 8, 4  # her 2-8, run by load (docs/orchestrator.md)


def idle_line():
    """The idle nudge, or None when 8+ agents run, or 2+ run and under 4 cores are free. Reads files only when called."""
    try:
        live = json.loads((TW / "notes" / "board-live.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        live = {}
    agents = live.get("agents") or []
    free = cap = None
    try:
        r = subprocess.run([sys.executable, str(ROOT / "cores.py"), "list"], cwd=str(ROOT), capture_output=True,
                           text=True, timeout=20, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        m = re.search(r"of cap (\d+).*?free (\d+)", r.stdout)
        if m:
            cap, free = int(m.group(1)), int(m.group(2))
    except (OSError, subprocess.SubprocessError):
        pass
    if len(agents) >= MAX_AGENTS or (len(agents) >= MIN_AGENTS and (free is None or free < MIN_FREE)):
        return None
    onboard = {a.get("task") for a in agents}
    try:
        text = (TW / "TASKS.md").read_text(encoding="utf-8")
    except OSError:
        text = ""
    open_t, ranked = set(), []
    for l in text.splitlines():
        m = HEAD.match(l)
        if not m:
            continue
        tid, pr, rest = m.groups()
        if pr in ("P1", "P2", "P3") and "(taking" not in rest and "{waits:" not in rest and "{after:" not in rest:
            ranked.append((pr, len(ranked), tid))
        open_t.add(tid)
    ready = {t for _, _, t in ranked}
    order = [t for t in live.get("upnext") or [] if t in ready]
    order += [t for _, _, t in sorted(ranked)]
    start = []
    for t in order:
        if t not in onboard and t not in start:
            start.append(t)
    start = start[:5]
    cores = f"{free} of {cap} cores free" if free is not None else "cores unknown"
    return f"idle: running {len(agents)} agents, {cores}, startable: {', '.join(start) or 'none found'}"


def disk_check(dw):
    """The disk line, if free space crossed a level since the last check; chimes for Anita as it falls."""
    try:
        r = dw.check()
    except OSError:
        return None
    if not r:
        return None
    text, falling, free = r
    if falling:
        disk.notify(free, falling)
    return text


def watch(path, idle, poll, box=None, session=None, dw=None):
    # At start, leftovers from before count as ready at once if she had pressed Send now.
    # The orchestrator handling part of a batch is not news; only actions not yet reported are.
    reported, seen_sent = set(), None
    told, next_reports = load_told(session), 0.0   # kept on disk so a restart doesn't reprint reports
    next_idle, last_idle = time.time() + IDLE_EVERY, None
    next_disk = 0.0
    while True:
        nudge = False
        if dw is not None and time.time() >= next_disk:
            next_disk = time.time() + DISK_EVERY
            l = disk_check(dw)
            if l:
                print(l, flush=True)
        if box is not None and time.time() >= next_reports:
            next_reports = time.time() + 10
            lines = agent_reports(box, session, told)
            if lines is None:
                print(f"session {box.current()[:8]} orchestrates trainworld now; this watch stops", flush=True)
                return
            for l in lines:
                print(l, flush=True)
            nudge = bool(lines)
        if box is not None and (nudge or time.time() >= next_idle):
            if time.time() >= next_idle:
                last_idle = None
            next_idle = time.time() + IDLE_EVERY
            l = idle_line()
            if l and l != last_idle:
                print(l, flush=True)
                last_idle = l
        pending, last, sent = state(path)
        pushed = sent is not None and sent != seen_sent
        seen_sent = sent
        if not pending:
            reported = set()
        else:
            ids = [r["id"] for r in pending]
            quiet = last is not None and time.time() - last >= idle
            pads = {r["id"] for r in pending if r.get("type") == "pad-send"} - reported
            if pushed or pads or (quiet and set(ids) - reported):
                why = "send now" if pushed else ("scratch pad" if pads else (f"idle {idle // 60} min" if idle >= 60 else f"idle {idle} s"))
                print(f"batch ready ({why}): {len(ids)} pending: {', '.join(ids)}", flush=True)
                reported |= set(ids)
        time.sleep(poll)


def append(path, obj):
    obj["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="watch", choices=["watch", "pending", "handled", "idle"])
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--by", help="who handled them (the orchestrator's name)")
    ap.add_argument("--note", default="", help="what was done, a few words")
    ap.add_argument("--idle", type=int, default=180)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--inbox", default=str(INBOX))
    ap.add_argument("--session", help="this orchestrator's session id: registers it for agent reports")
    ap.add_argument("--root", default=str(TW), help="the trainworld/ whose agent reports to watch (for testing)")
    ap.add_argument("--disk-warn-gb", type=float, default=disk.WARN_GB, help="free space that prints the low line")
    ap.add_argument("--disk-urgent-gb", type=float, default=disk.URGENT_GB, help="free space that prints the urgent line")
    a = ap.parse_args()
    path = Path(a.inbox)
    if a.cmd == "watch":
        box = reports.Box(a.root)
        session = a.session or box.current()     # without --session, whoever orchestrates as it starts
        if a.session:
            box.register(a.session)
        try:
            watch(path, a.idle, a.poll, box, session, disk.Watch(a.disk_warn_gb, a.disk_urgent_gb))
        except KeyboardInterrupt:
            pass
    elif a.cmd == "idle":
        print(idle_line() or "busy")
    elif a.cmd == "pending":
        for r in state(path)[0]:
            print(json.dumps(r, ensure_ascii=False))
    else:
        if not a.ids or not a.by:
            sys.exit("handled needs ids and --by")
        known = {str(r["id"]) for r in state(path)[0]}
        unknown = [i for i in a.ids if i not in known]
        if unknown:
            sys.exit(f"not pending, nothing marked: {', '.join(unknown)}")
        for i in a.ids:
            append(path, {"handled": i, "by": a.by, "note": a.note})
        print(f"marked {len(a.ids)} handled")


if __name__ == "__main__":
    main()
