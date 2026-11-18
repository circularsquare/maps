"""PreToolUse hook (T-3665): refuse Edit/Write to the shared bookkeeping docs inside an agent tree.

TASKS.md, DECISIONS.md, ARCHIVE.md and LINES.md are edited in the main tree only; an edit made under
maps/.tw-trees/<name>/trainworld/ lands as a rebase conflict (27 such edits on 2026-11-17/18). Reads the hook JSON on
stdin; exit 2 with the reason on stderr refuses the call. `python tree_guard.py --test <path>` prints the verdict.
"""
import json
import re
import sys

DOCS = ("TASKS.md", "DECISIONS.md", "ARCHIVE.md", "LINES.md")
TREE = re.compile(r"^(.*?)[\\/]\.tw-trees[\\/]([^\\/]+)[\\/](trainworld[\\/](?:TASKS|DECISIONS|ARCHIVE|LINES)\.md)$", re.I)


def verdict(path):
    """None when the path is fine, else the message."""
    m = TREE.match(path or "")
    if not m:
        return None
    main = m.group(1).replace("\\", "/") + "/" + m.group(3).replace("\\", "/")
    return ("%s is edited in the main tree only, not in agent tree %s (a copy here conflicts when the tree lands). "
            "Edit %s instead, re-reading it first; for tasks use tools/tasks.py." % (m.group(3).split("/")[-1].split("\\")[-1], m.group(2), main))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--test":
        print(verdict(sys.argv[2]) or "ok")
        return 0
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0                                   # never block on a malformed call
    ti = data.get("tool_input") or {}
    msg = verdict(ti.get("file_path") or ti.get("notebook_path") or "")
    if msg:
        sys.stderr.write(msg + "\n")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
