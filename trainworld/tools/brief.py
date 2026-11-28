"""The orchestrator's brief for a code agent, about 15 lines, from where the facts already live (T-4421). Run from the maps root:
  venv\Scripts\python.exe trainworld/tools/brief.py <task id> --kind <row> --model <m> --tree <name> --port <n> --ids <ids>
      [--decisions "..."] [--todo "<text in the TODO.md short-term item>"] [--out <file>]
--kind: a row of the measuring budget table in trainworld/CLAUDE.md (ui, logic, edit, sim, loading, or any part of the row's
name), quoted exactly; "tooling" for work with no game runs. --ids: task ids the agent may file ("none"). --decisions may
repeat. --todo copies the matching short-term item of TODO.md word for word, its indented lines too.
Reads TASKS.md, notes, TODO.md and the board from the main tree (a worktree has no board); pastes no task block.
--root <dir> reads another copy of trainworld/ (for testing)."""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KINDS = {"ui": "UI, text, CSS", "logic": "App logic", "edit": "Edit speed", "sim": "Trip figures, sim", "loading": "Loading, memory"}


def main_root(root):
    """The first worktree's trainworld/ (the main tree), where TASKS.md, TODO.md and the board are current."""
    try:
        wl = subprocess.run(["git", "-C", os.path.dirname(root), "worktree", "list", "--porcelain"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=8).stdout
        first = next(l[9:] for l in wl.split("\n") if l.startswith("worktree "))
        return os.path.join(first, "trainworld")
    except (OSError, subprocess.SubprocessError, StopIteration):
        return root


def text(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().replace("\r\n", "\n")
    except OSError:
        return ""


def budget_row(root, kind):
    if kind.lower() == "tooling":
        return "Tooling", "no game runs, no benchmarks"
    rows = []
    for l in text(os.path.join(root, "CLAUDE.md")).split("\n"):
        l = l.strip()
        cells = [c.strip() for c in l.strip("|").split("|")]
        if l.startswith("|") and len(cells) == 2 and not set(l) <= set("|- ") and cells[0] != "Kind of change":
            rows.append(cells)
    k = KINDS.get(kind.lower(), kind).lower()
    hit = [r for r in rows if k in r[0].lower()]
    if len(hit) != 1:
        sys.exit(f"--kind {kind!r} matches {len(hit)} budget rows; rows: " + "; ".join(r[0] for r in rows) + "; or tooling")
    return hit[0]


def task_info(live, tid, out_root):
    title, other = "", ""
    for l in text(os.path.join(live, "TASKS.md")).split("\n"):
        if l.startswith(f"- {tid} ["):
            title = re.sub(r"\{[^}]*\}\s*|\(taking[^)]*\)\s*", "", l.split("] ", 1)[1]).strip()
            i = l and text(os.path.join(live, "TASKS.md")).split("\n").index(l) + 1
            for m in text(os.path.join(live, "TASKS.md")).split("\n")[i:]:
                if not m.startswith(" "):
                    break
                if m.strip().startswith("- Other:"):
                    other = m
            break
    if not title:
        sys.exit(f"{tid} is not in TASKS.md")
    own = f"notes/{tid}.md"
    if os.path.exists(os.path.join(live, own)):
        return title, own
    m = re.search(r"notes/[\w.-]+\.md", other)
    return title, (m.group(0) if m else f"{own} (new)")


def todo_item(live, needle):
    lines = text(os.path.join(live, "TODO.md")).split("\n")
    start = next((i for i, l in enumerate(lines) if "short term" in l.lower()), None)
    if start is None:
        sys.exit("TODO.md has no short term section")
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("-----")), len(lines))
    items, cur = [], []
    for l in lines[start + 1:end]:
        if l.strip() and not l.startswith((" ", "\t")) and cur:
            items.append(cur)
            cur = []
        if l.strip() or cur:
            cur.append(l)
    if cur:
        items.append(cur)
    hit = [i for i in items if needle.lower() in "\n".join(i).lower()]
    if len(hit) != 1:
        sys.exit(f"--todo {needle!r} matches {len(hit)} short-term items")
    return "\n".join(hit[0]).rstrip()


def others(live, me):
    """Other agents on the board, with the dirty files of their trees."""
    try:
        board = json.loads(text(os.path.join(live, "notes", "board-live.json")))
    except ValueError:
        board = {}
    main = os.path.dirname(live)
    out = []
    for a in board.get("agents") or []:
        if a.get("tree") == me or a["name"] == me:
            continue
        tree = a.get("tree")
        s = f"{a['name']} {a.get('task', '')}"
        if tree:
            s += f" (tree {tree}"
            try:
                st = subprocess.run(["git", "-C", os.path.join(main, ".tw-trees", tree), "status", "--porcelain", "-uno"],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=8).stdout
                files = [l[3:].strip('"') for l in st.split("\n") if l.strip()]
                if files:
                    s += ": " + ", ".join(files[:6]) + (f" and {len(files) - 6} more" if len(files) > 6 else "")
            except (OSError, subprocess.SubprocessError):
                pass
            s += ")"
        out.append(s)
    return out


def scratch(live, tree):
    try:
        orch = json.loads(text(os.path.join(live, "notes", "board-live.json"))).get("orchestrator", "").split()[0]
    except (ValueError, IndexError):
        orch = ""
    hits = glob.glob(os.path.join(tempfile.gettempdir(), "claude", "*", orch + "*", "scratchpad")) if orch else []
    base = hits[0] if hits else "<your scratchpad>"
    return os.path.join(base, tree).replace("\\", "/")


def similar(live, kind, tid):
    """One line on what finished agents of the same budget row (and the task's LINES line, if 3+ share it) used, from
    notes/agent-tokens.jsonl (T-4600; her 11-28: 200-500k tokens an agent, a note at 400k)."""
    recs = []
    for l in text(os.path.join(live, "notes", "agent-tokens.jsonl")).split("\n"):
        try:
            recs.append(json.loads(l))
        except ValueError:
            pass
    pool = [r for r in recs if r.get("kind") == kind and r.get("task") != tid]
    line = ""
    m = re.search(r"\{L:([\w-]+)\}", next((l for l in text(os.path.join(live, "TASKS.md")).split("\n") if l.startswith(f"- {tid} [")), ""))
    if m and len([r for r in pool if r.get("line") == m.group(1)]) >= 3:
        pool = [r for r in pool if r.get("line") == m.group(1)]
        line = f" on line {m.group(1)}"
    if len(pool) < 3:
        return f"Size: {len(pool)} finished {kind} agent(s) on record, too few for a figure; aim for one context of 200-500k tokens."
    v = sorted(r["tokens"] for r in pool)
    over = sum(x > 500_000 for x in v)
    med = v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) // 2
    s = (f"Size: {len(v)} finished {kind} agents{line} used a median {med // 1000}k tokens, highest {v[-1] // 1000}k"
         f" (target 200-500k; you get a note at 400k).")
    return s + (f" {over} of {len(v)} ran past 500k: split it." if over * 3 >= len(v) else "")


def build(a, root):
    live = main_root(root)
    title, notes = task_info(live, a.task, root)
    kind, row = budget_row(root, a.kind)
    tree_path = f"C:/Users/anita/projects/maps/.tw-trees/{a.tree}"
    L = [f"Task {a.task}: {title}. Grep trainworld/TASKS.md for {a.task}; notes: {notes}.",
         f"Kind: {kind}. Budget, from the table in trainworld/CLAUDE.md: {kind} | {row}. Add no measuring beyond it.",
         "Start from the code (CLAUDE.md \"Read before you run\"). Read trainworld/CLAUDE.md first; it has the rules, which this brief doesn't repeat."]
    L.append(similar(live, kind, a.task))
    for d in a.decisions or []:
        L.append(f"Decision: {d}")
    if a.todo:
        L.append("From TODO.md, verbatim:\n" + todo_item(live, a.todo))
    L.append(f"Work only in the worktree {tree_path} (run everything from it).")
    o = others(live, a.tree)
    L.append("Other agents are working at the same time; leave their trees and files alone: " + ("; ".join(o) if o else "none") + ".")
    L.append(f"Your scratchpad folder: {scratch(live, a.tree)}. Vite port {a.port}: node trainworld/checks/dev-nowatch.mjs {a.port}.")
    L.append(f"Task ids you may file: {a.ids}. Run tasks.py start/done from the maps root yourself, with --agent {a.tree}. Don't commit.")
    L.append(f"Report in at most 15 lines; detail goes in notes/{a.task}.md. Stop your Vite and any background command before your final report.")
    return "\n".join(L) + "\n"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    ap.add_argument("--kind", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--port", required=True, type=int)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--decisions", action="append")
    ap.add_argument("--todo")
    ap.add_argument("--out")
    ap.add_argument("--root", default=os.path.dirname(HERE))
    a = ap.parse_args()
    brief = build(a, a.root)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(brief)
        print(f"{a.out}: {len(brief)} characters, {len(brief.splitlines())} lines, for {a.model}")
    else:
        print(brief, end="")


main()
