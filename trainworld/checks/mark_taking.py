"""Mark TASKS.md lines as taken: python mark_taking.py <tasks.md> <id>=<agent> ..."""
import re
import sys
from datetime import date

path, *pairs = sys.argv[1:]
with open(path, encoding="utf-8", newline="") as f:
    lines = f.read().split("\n")
for pair in pairs:
    tid, agent = pair.split("=", 1)
    for i, line in enumerate(lines):
        if line.startswith(f"- {tid} ["):
            body = line.rstrip("\r")
            end = "\r" if line.endswith("\r") else ""
            if body.endswith(" Open."):
                body = body[: -len(".")] + ";"
            lines[i] = f"{body} taking, {date.today().isoformat()}, agent {agent}.{end}"
            print(f"{tid}: marked")
            break
    else:
        print(f"{tid}: not found")
with open(path, "w", encoding="utf-8", newline="") as f:
    f.write("\n".join(lines))
