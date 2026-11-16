"""The orchestrator's side of the board inbox (trainworld/notes/board-inbox.jsonl).

    python trainworld/board/watch.py              # keeps running; one line per batch that is ready
    python trainworld/board/watch.py pending      # her pending actions, one JSON line each
    python trainworld/board/watch.py handled <id> [<id> ...] --by <orchestrator> [--note "..."]

A batch is ready when something is pending and she pressed Send now after the last batch was
reported, or a pad-send (the scratch pad's Send for filing) is new, or nothing new has come from
her for --idle seconds (180). The watch prints only then,
for the Monitor tool. The line format and the action types are in notes/T-3391.md.
"""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

INBOX = Path(__file__).resolve().parents[1] / "notes" / "board-inbox.jsonl"


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


def watch(path, idle, poll):
    # At start, leftovers from before count as ready at once if she had pressed Send now.
    # The orchestrator handling part of a batch is not news; only actions not yet reported are.
    reported, seen_sent = set(), None
    while True:
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
    ap.add_argument("cmd", nargs="?", default="watch", choices=["watch", "pending", "handled"])
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--by", help="who handled them (the orchestrator's name)")
    ap.add_argument("--note", default="", help="what was done, a few words")
    ap.add_argument("--idle", type=int, default=180)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--inbox", default=str(INBOX))
    a = ap.parse_args()
    path = Path(a.inbox)
    if a.cmd == "watch":
        try:
            watch(path, a.idle, a.poll)
        except KeyboardInterrupt:
            pass
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
