"""The orchestrator's live state, read from where it already lives (T-3667). Read-only; run from the maps root:
  venv\Scripts\python.exe trainworld/tools/handoff.py status
Prints: master and origin, 8800's build, worktrees under .tw-trees (branch, commits ahead, dirty files, last change,
a process holding it), agents on the board with start times, uncommitted trainworld paths in the main tree, open asks,
unread agent reports (tools/reports.py), core claims, a running world run, the unpublished changelog count. notes/handoff.md keeps only the plan, her words
and judgement; this prints the rest.
--root <dir> reads another copy of trainworld/ (for testing)."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, cwd=None, timeout=8):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        return p.stdout.rstrip() if p.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git(cwd, *args):
    return run(["git", "-C", cwd, *args])


def jload(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def parse_t(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def ago(secs):
    secs = int(secs)
    if secs < 90:
        return f"{secs}s"
    if secs < 5400:
        return f"{secs // 60} min"
    if secs < 172800:
        return f"{secs / 3600:.1f} h"
    return f"{secs // 86400} d"


def since(ts):
    d = ts if isinstance(ts, datetime) else parse_t(ts)
    if d is None:
        return "?"
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return ago((datetime.now(timezone.utc) - d).total_seconds())


def process_lines():
    """Lower-cased command lines of every process, one PowerShell call; empty when it is slow or fails."""
    out = run(["powershell", "-NoProfile", "-Command",
               "Get-CimInstance Win32_Process | ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }"], timeout=6)
    return [l.lower().replace("\\", "/") for l in (out or "").split("\n") if l.strip()]


def section_git(main, out):
    head = git(main, "rev-parse", "--short", "HEAD")
    branch = git(main, "rev-parse", "--abbrev-ref", "HEAD")
    ahead = git(main, "rev-list", "--count", "origin/master..master")
    behind = git(main, "rev-list", "--count", "master..origin/master")
    note = ""
    if ahead is not None:
        note = "pushed" if ahead == "0" and behind == "0" else f"{ahead} ahead of origin, {behind} behind (as of the last fetch)"
    out.append(f"master {head} ({branch}); {note or 'origin unknown'}")
    return head


def section_builds(board, head, out):
    b = (board or {}).get("builds") or {}
    live = b.get("8800")
    if not live:
        out.append("8800: no build recorded on the board")
        return
    n = git(os.path.dirname(HERE), "rev-list", "--count", f"{live}..master") if live != head else "0"
    state = "= master" if live == head else (f"{n} commits behind master" if n not in (None, "") else "not an ancestor of master?")
    out.append(f"8800 = {live} ({state}); board says master {b.get('master')}, live {b.get('live')}")


def section_trees(main, board, out):
    wl = git(main, "worktree", "list", "--porcelain") or ""
    trees = []
    cur = {}
    for l in wl.split("\n") + [""]:
        if not l.strip():
            if cur.get("worktree"):
                trees.append(cur)
            cur = {}
        else:
            k, _, v = l.partition(" ")
            cur[k] = v
    trees = [t for t in trees if "/.tw-trees/" in t["worktree"].replace("\\", "/")]
    if not trees:
        out.append("no worktrees under .tw-trees")
        return
    agents_by_tree = {}
    for a in (board or {}).get("agents") or []:
        agents_by_tree.setdefault(a.get("tree"), []).append(a["name"])
    procs = process_lines()
    for t in trees:
        path = t["worktree"]
        name = os.path.basename(path)
        branch = t.get("branch", "(detached)").replace("refs/heads/", "")
        ahead = git(path, "rev-list", "--count", "master..HEAD") or "?"
        behind = git(path, "rev-list", "--count", "HEAD..master") or "?"
        dirty = (git(path, "status", "--porcelain", "-uno") or "").split("\n")
        dirty = [d for d in dirty if d.strip()]
        newest = float(git(path, "log", "-1", "--format=%ct") or 0)
        for d in dirty:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(path, d[3:].split(" -> ")[-1].strip('"'))))
            except OSError:
                pass
        key = path.replace("\\", "/").lower()
        held = sum(1 for p in procs if key in p and "handoff.py" not in p)
        who = ",".join(agents_by_tree.get(name, [])) or "no agent on the board"
        out.append(f"  {name} [{branch}] +{ahead}/-{behind} vs master, {len(dirty)} dirty, changed {ago(time.time() - newest) if newest else '?'} ago"
                   f"{f', {held} process(es)' if held else ''}; {who}")


def section_agents(board, out):
    ags = (board or {}).get("agents") or []
    if not ags:
        out.append("  none")
    for a in ags:
        out.append(f"  {a['name']} ({a.get('model', '?')}) {a.get('task', '')}: {a.get('what', '')}; started {since(a.get('started'))} ago")
    runs = (board or {}).get("runs") or []
    if runs:
        out.append(f"  runs on the board: {json.dumps(runs, ensure_ascii=False)[:200]}")
    q = (board or {}).get("upnext") or []
    if q:
        out.append("  up next: " + ", ".join(q))


def section_uncommitted(main, out):
    st = git(main, "status", "--porcelain", "--", "trainworld", "serve.py") or ""
    groups = {}
    for l in st.split("\n"):
        if not l.strip():
            continue
        code, p = l[:2].strip() or "M", l[3:].strip('"')
        parts = p.split("/")
        if parts[0] == "trainworld" and len(parts) > 2:
            g = "/".join(parts[:2]) + ("/" if len(parts) > 2 else "")
        else:
            g = p
        groups.setdefault(g, []).append(code)
    if not groups:
        out.append("  clean")
        return
    items = []
    for g, codes in sorted(groups.items()):
        n = len(codes)
        items.append(g if n == 1 else f"{g} x{n}")
        if codes.count("??") == n and n == 1:
            items[-1] += " (new)"
    line = "  "
    for it in items:
        if len(line) + len(it) > 110:
            out.append(line.rstrip(", "))
            line = "  "
        line += it + ", "
    out.append(line.rstrip(", "))


def section_asks(root, out):
    asks = jload(os.path.join(root, "notes", "asks.json"))
    if not asks:
        out.append("  none")
        return
    for a in asks:
        try:
            age = f"{(date.today() - date.fromisoformat(a.get('since', ''))).days}d"
        except ValueError:
            age = "?"
        out.append(f"  {a.get('id')} ({age}, {a.get('by', '')}) {a.get('title') or a.get('text', '')[:70]}")


def section_reports(root, out):
    sys.path.insert(0, HERE)
    import reports
    box = reports.Box(root)
    cur = box.current()
    rs = box.unread()
    theirs = [r for r in rs if not (cur and str(r.get("session", "")).startswith(cur[:8]))]
    for r in theirs:
        out.append("  " + reports.line(r, 116))
    mine = len(rs) - len(theirs)
    if mine:
        out.append(f"  {mine} more from the current orchestrator's own agents (`reports.py list`)")
    if not rs:
        out.append("  none")
    elif theirs:
        out.append("  `reports.py show <id>` to read; `reports.py read <id> --by <you>` once acted on")


def section_cores(main, out):
    r = run([sys.executable, os.path.join(main, "cores.py"), "list"], cwd=main)
    lines = [l for l in (r or "cores.py list failed").split("\n") if l.strip()]
    for l in lines:
        out.append("  " + l)


def section_world_run(main, out):
    wr = os.path.join(main, "trainworld", "pipeline", "world_run.py")
    # status prints a line per shipped row; a failing exit is fine, we read what came
    try:
        p = subprocess.run([sys.executable, "-I", wr, "status"], cwd=main, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=10)
        text = p.stdout or ""
    except (OSError, subprocess.SubprocessError):
        text = ""
    keep = []
    for l in text.split("\n"):
        if l.startswith(("STOP file", "{", "lock ")) or "claimed by run" in l:
            keep.append(l)
    if not keep:
        out.append("  world_run status gave nothing")
        return
    stop = keep[0]
    locks = [l for l in keep if l.startswith(("lock ", "  ")) and ("claimed by" in l or l.startswith("lock "))]
    counts = next((l for l in keep if l.startswith("{")), "")
    out.append(f"  rows {counts}; " + ("STOP file present" if "(present" in stop else "no STOP file"))
    if locks:
        for l in locks[:6]:
            out.append("  " + l.strip())
    else:
        out.append("  no run holds a lock or claims a row")


def section_changelog(root, out):
    d = jload(os.path.join(root, "changelog", "next.json")) or {}
    items = d.get("items") or []
    hidden = sum(1 for i in items if i.get("hidden"))
    out.append(f"  {len(items) - hidden} unpublished items ({hidden} hidden); `changelog.py list` shows them")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(HERE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    a = ap.parse_args()
    root = a.root
    main_tree = os.path.dirname(root)
    wl = git(main_tree, "worktree", "list", "--porcelain") or ""
    first = next((l[9:] for l in wl.split("\n") if l.startswith("worktree ")), main_tree)
    main_tree = first
    live_root = os.path.join(main_tree, "trainworld")
    board = jload(os.path.join(live_root, "notes", "board-live.json"))
    t0 = time.time()
    out = [f"handoff status {datetime.now().strftime('%Y-%m-%d %H:%M')} local (orchestrator on the board: {(board or {}).get('orchestrator', '?')})"]
    head = section_git(main_tree, out)
    section_builds(board, head, out)
    out.append("trees (.tw-trees):")
    section_trees(main_tree, board, out)
    out.append("agents on the board:")
    section_agents(board, out)
    out.append("uncommitted in the main tree (trainworld, serve.py):")
    section_uncommitted(main_tree, out)
    out.append("open asks:")
    section_asks(live_root, out)
    out.append("unread agent reports (from sessions other than the current orchestrator):")
    section_reports(live_root, out)
    out.append("core claims:")
    section_cores(main_tree, out)
    out.append("world run:")
    section_world_run(main_tree, out)
    out.append("changelog:")
    section_changelog(live_root, out)
    out.append(f"({time.time() - t0:.1f} s)")
    print("\n".join(out))


main()
