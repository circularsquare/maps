"""Tokens per finished agent (T-4600): one line each in notes/agent-tokens.jsonl, so tasks are sized from what similar
ones used (Anita 2026-11-28: 200-500k tokens per agent context, a note at 400k). Run from the maps root:

  venv\Scripts\python.exe trainworld/tools/agent_tokens.py sync [--all]   add agents of the orchestrating sessions
                                                                          (--all: also ones still running)
  venv\Scripts\python.exe trainworld/tools/agent_tokens.py stats          median and highest per budget row
  venv\Scripts\python.exe trainworld/tools/agent_tokens.py add <transcript or agent id> [--root <dir>]

A record: task, kind (the budget row named in the brief's "Kind:" line; "unknown" for older briefs), model, tokens (the
largest context an agent reached: input + cache + output of one call, the figure the 400k note watches), final
(its last call's), calls, minutes (first to last entry), compacted, agent_id, session, name, at.
reports.py's SubagentStop hook calls add_from_stop, so the file stays current without a sweep.
--root <dir> reads and writes another copy of trainworld/ (for testing)."""
import argparse
import glob
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import claude2  # noqa: E402
import reports  # noqa: E402

USAGE = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")
KIND = re.compile(r"^\s*Kind:\s*([^.\n]+)", re.M)
ROWS = ("UI, text, CSS", "App logic", "Edit speed", "Trip figures, sim", "Loading, memory", "Tooling")


def kind_of(prompt):
    """The budget row a brief names in its "Kind:" line (brief.py writes the row's name; "tooling" for no game runs)."""
    for m in KIND.finditer(prompt):
        k = m.group(1).strip().lower()
        for row in ROWS:
            if k.startswith(row.lower()):
                return row
    return "unknown"
IDLE_MINUTES = 10          # an unreported agent whose transcript is this old counts as finished


def file_path(root):
    return os.path.join(root, "notes", "agent-tokens.jsonl")


def measure(path):
    """(record fields from the transcript, or None if it holds no model call)."""
    es = reports.entries(path)
    ctx, models, stamps = [], {}, []
    seen = set()
    for e in es:
        ts = e.get("timestamp")
        if ts:
            stamps.append(ts)
        msg = e.get("message") if e.get("type") == "assistant" else None
        u = (msg or {}).get("usage")
        if not isinstance(u, dict):
            continue
        key = msg.get("id") or e.get("uuid")
        total = sum(u.get(k) or 0 for k in USAGE)
        if not total:
            continue                             # an entry the harness wrote without a model call
        if key in seen:
            ctx[-1] = total                      # the same reply written again with its final usage
            continue
        seen.add(key)
        ctx.append(total)
        models[msg.get("model", "?")] = models.get(msg.get("model", "?"), 0) + 1
    if not ctx:
        return None
    t0, t1 = reports.parse_t(stamps[0]), reports.parse_t(stamps[-1])
    minutes = round((t1 - t0).total_seconds() / 60, 1) if t0 and t1 else None
    drops = sum(1 for a, b in zip(ctx, ctx[1:]) if b < a * 0.6 and a > 50_000)
    return {"model": max(models, key=models.get), "tokens": max(ctx), "final": ctx[-1], "calls": len(ctx),
            "minutes": minutes, "compacted": drops > 0, "start": stamps[0], "at": stamps[-1]}, es


def task_line(root, tid):
    """The {L:slug} of a task, from TASKS.md or the archive; '' if not found."""
    for name in ("TASKS.md", "ARCHIVE.md"):
        try:
            with open(os.path.join(root, name), encoding="utf-8", errors="replace") as f:
                for l in f:
                    if l.startswith(f"- {tid} ") or l.startswith(f"- {tid}\t"):
                        m = re.search(r"\{L:([\w-]+)\}", l)
                        if m:
                            return m.group(1)
        except OSError:
            pass
    return ""


def make(root, path, session=None):
    """The record for one agent transcript, or None (an agent's helper, or no model call)."""
    meta = reports.meta_for(path)
    if int(meta.get("spawnDepth") or 1) > 1:
        return None
    got = measure(path)
    if not got:
        return None
    m, es = got
    first = reports.first_prompt_text(es)
    desc = meta.get("description", "")
    task = (reports.TASK.search(desc) or reports.TASK.search(first[:400]) or [None])[0]
    name = reports.AGENT_NAME.search(first[:2000])
    box = reports.Box(root)
    agent_id = os.path.basename(path)[6:-6]
    if not session:
        session = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return {"task": task, "kind": kind_of(first), "line": task_line(root, task) if task else "",
            "model": meta.get("model") or m["model"], "tokens": m["tokens"], "final": m["final"], "calls": m["calls"],
            "minutes": m["minutes"], "compacted": m["compacted"], "agent_id": agent_id, "session": session,
            "name": (name.group(1) if name else None) or box.board_name(task) or desc[:40], "start": m["start"],
            "at": m["at"], "size": os.path.getsize(path)}


def load(root):
    out = {}
    try:
        with open(file_path(root), encoding="utf-8") as f:
            for l in f:
                try:
                    r = json.loads(l)
                    out[r["agent_id"]] = r
                except (ValueError, KeyError):
                    pass
    except OSError:
        pass
    return out


def save(root, recs):
    rows = sorted(recs.values(), key=lambda r: r.get("start") or "")
    os.makedirs(os.path.dirname(file_path(root)), exist_ok=True)
    tmp = file_path(root) + f".{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, file_path(root))


def upsert(root, rec):
    recs = load(root)
    recs[rec["agent_id"]] = rec
    save(root, recs)


def add_from_stop(box, data):
    """SubagentStop: keep this agent's record when the session orchestrates trainworld and the agent is top-level."""
    sid = data.get("session_id") or ""
    path = data.get("agent_transcript_path") or ""
    if not sid or not path or not box.accepts(sid):
        return None
    rec = make(box.root, path, sid)
    if rec:
        upsert(box.root, rec)
    return rec


def sync(root, everything=False):
    import time
    box = reports.Box(root)
    recs, new = load(root), 0
    for sid in box.registered():
        for path in [p for d in claude2.project_dirs() for p in glob.glob(os.path.join(d, "*", sid, "subagents", "agent-*.jsonl"))]:
            aid = os.path.basename(path)[6:-6]
            if not everything and time.time() - os.path.getmtime(path) < IDLE_MINUTES * 60:
                continue
            if aid in recs and recs[aid].get("size") == os.path.getsize(path):
                continue
            rec = make(root, path, sid)
            if rec and recs.get(aid) != rec:
                new += aid not in recs
                recs[aid] = rec
    save(root, recs)
    return len(recs), new


def groups(recs):
    by = {}
    for r in recs:
        by.setdefault(r["kind"], []).append(r["tokens"])
    return by


def mins(kind, recs):
    m = [r["minutes"] for r in recs if r["kind"] == kind and r.get("minutes") is not None]
    return int(statistics.median(m)) if m else "?"


def stats(root):
    recs = list(load(root).values())
    for kind, v in sorted(groups(recs).items(), key=lambda kv: -len(kv[1])):
        print(f"{kind:28} n={len(v):3}  median {int(statistics.median(v)) // 1000}k  highest {max(v) // 1000}k  "
              f"past 500k: {sum(x > 500_000 for x in v)}  median minutes {mins(kind, recs)}")
    print(f"{len(recs)} agents, median {int(statistics.median(r['tokens'] for r in recs)) // 1000}k" if recs else "no records")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(HERE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sync")
    s.add_argument("--all", action="store_true")
    sub.add_parser("stats")
    a = sub.add_parser("add")
    a.add_argument("transcript")
    ns = ap.parse_args()
    if ns.cmd == "sync":
        n, new = sync(ns.root, ns.all)
        print(f"{n} records, {new} new")
    elif ns.cmd == "stats":
        stats(ns.root)
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(HERE), "checks"))
        import token_usage
        path = token_usage.find(ns.transcript)
        rec = make(ns.root, path)
        if not rec:
            sys.exit("no record: an agent's helper, or no model call")
        upsert(ns.root, rec)
        print(json.dumps(rec, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
