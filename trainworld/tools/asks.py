"""Open asks by command (T-3667). Run from the maps root:
  venv\Scripts\python.exe trainworld/tools/asks.py add --title "few words" --text "the whole question" --by <name>
  venv\Scripts\python.exe trainworld/tools/asks.py answer <id> --note "what she said, or why it closed" [--by <name>]
  venv\Scripts\python.exe trainworld/tools/asks.py list [--full]
notes/asks.json is the board's list of asks that are not tasks: [{id, title, text, since, by}], oldest first.
Ids are a-YYYY-MM-DD-N and are never reused: every add and answer is also appended to notes/asks-log.jsonl, and the
next N for a day is one past the highest seen in the file or the log. Nothing else edits asks.json.
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

ID = re.compile(r"^a-(\d{4}-\d{2}-\d{2})-(\d+)$")


class Store:
    def __init__(self, root):
        self.file = os.path.join(root, "notes", "asks.json")
        self.log = os.path.join(root, "notes", "asks-log.jsonl")
        self.lock = os.path.join(root, "notes", "asks.lock")

    def asks(self):
        if not os.path.exists(self.file):
            return []
        with open(self.file, encoding="utf-8") as f:
            raw = f.read().strip()
        data = json.loads(raw) if raw else []
        if not isinstance(data, list):
            raise SystemExit("asks.json is not a list; fix it by hand once")
        return data

    def save(self, asks):
        tmp = self.file + ".tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(asks, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp, self.file)

    def logged(self):
        out = []
        if os.path.exists(self.log):
            with open(self.log, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            out.append(json.loads(line))
                        except ValueError:
                            pass
        return out

    def append(self, rec):
        with open(self.log, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def next_id(self, asks, day):
        top = 0
        for i in [a.get("id", "") for a in asks] + [e.get("id", "") for e in self.logged()]:
            m = ID.match(str(i))
            if m and m.group(1) == day:
                top = max(top, int(m.group(2)))
        return f"a-{day}-{top + 1}"


def locked(path):
    return FileLock(path, err=SystemExit, message=f"{path} is held; if no asks.py is running, delete it")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cmd_add(s, a):
    with locked(s.lock):
        asks = s.asks()
        day = date.today().isoformat()
        aid = s.next_id(asks, day)
        entry = {"id": aid, "title": a.title.strip(), "text": a.text.strip(), "since": day, "by": a.by}
        asks.append(entry)
        s.append({"event": "add", "at": now(), **entry})
        s.save(asks)
    print(f"{aid}: added ({len(asks)} open)")
    return 0


def cmd_answer(s, a):
    with locked(s.lock):
        asks = s.asks()
        hit = next((x for x in asks if x.get("id") == a.id), None)
        if hit is None:
            print(f"{a.id}: not an open ask (answered already, or a wrong id); `asks.py list` shows the open ones")
            return 1
        asks = [x for x in asks if x is not hit]
        s.append({"event": "answer", "at": now(), "id": a.id, "title": hit.get("title", ""), "note": a.note, "by": a.by or ""})
        s.save(asks)
    print(f"{a.id}: answered, removed ({len(asks)} open)")
    return 0


def age(since):
    try:
        d = (date.today() - date.fromisoformat(since)).days
    except ValueError:
        return "?"
    return "today" if d == 0 else f"{d}d"


def cmd_list(s, a):
    asks = s.asks()
    if not asks:
        print("no open asks")
        return 0
    for x in sorted(asks, key=lambda x: x.get("since") or "9"):
        print(f"{x.get('id')} ({age(x.get('since', ''))}, {x.get('by', '')}) {x.get('title') or x.get('text', '')[:70]}")
        if a.full:
            print(f"    {x.get('text', '')}")
    return 0


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(HERE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("add")
    p.add_argument("--title", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--by", required=True)
    p = sub.add_parser("answer")
    p.add_argument("id")
    p.add_argument("--note", required=True)
    p.add_argument("--by")
    p = sub.add_parser("list")
    p.add_argument("--full", action="store_true")
    a = ap.parse_args()
    s = Store(a.root)
    sys.exit({"add": cmd_add, "answer": cmd_answer, "list": cmd_list}[a.cmd](s, a))


main()
