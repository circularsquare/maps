"""Agent reports inbox (T-3668): each orchestrator subagent's final report is kept as a file, so whichever
orchestrator is current reads it, not only the session that started the agent. Run from the maps root:

  venv\Scripts\python.exe trainworld/tools/reports.py list [--all]          unread reports, one line each
  venv\Scripts\python.exe trainworld/tools/reports.py show <key> ...        full text (key: report id, T-id or agent name)
  venv\Scripts\python.exe trainworld/tools/reports.py read <key> ... | --session <id>  --by <name> [--note "..."]
                                                        mark read once acted on (tasks.py done/close does it for its T-id)
  venv\Scripts\python.exe trainworld/tools/reports.py sweep [--session <id>] [--since <ISO time>]
  venv\Scripts\python.exe trainworld/tools/reports.py hook                  the SubagentStop hook (stdin JSON)
  venv\Scripts\python.exe trainworld/tools/reports.py add --name <agent> --text-file <f> [--task T-1] [--kind gemini]
                                                        keep a report that no hook sees (gemini_agent.py calls Box.add)

The hook (registered in .claude/settings.json) fires when any subagent of any maps session stops. It keeps only
top-level agents of a session that orchestrates trainworld: one registered by `watch.py --session`, or named on
notes/handoff.md's Orchestrator or Previous line. The reports are the agent's successful SubagentHandback
messages and SendMessages to "main"; a spawned agent (one with a .meta.json) without a SubagentHandback tool reports by its last text, unless that is a compaction summary. It always
exits 0. It prints nothing, except once per agent stop: a final-text report of more than 18 non-empty lines
(REPORT_MAX_LINES) is answered with {"decision":"block","reason":...}, which makes the agent rewrite it in 15 or
fewer. The second stop (stop_hook_active, or an agent already in notes/agent-reports/blocked/) always passes, so
nothing loops. Reports delivered by SubagentHandback are already sent, so they are not measured.

Files: notes/agent-reports/<id>.json unread, read/<id>.json once marked; one file per report, written by temp file
and os.replace, so agents finishing together never clash and a repeat stop writes nothing new.
sweep reads the same transcripts after the fact, for reports a hook missed (handback and SendMessage only).
--root <dir> runs on another copy of trainworld/ (for testing)."""
import argparse
import glob
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = re.compile(r"\bT-\d+\b")
AGENT_NAME = re.compile(r"You are (?:the )?agent `([\w.-]+)`")
SESSION_LINE = re.compile(r"^(Orchestrator|Previous):\s*session\s+([0-9a-f-]{8,})", re.M)


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_t(s):
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class Box:
    def __init__(self, root):
        self.root = root
        self.dir = os.path.join(root, "notes", "agent-reports")
        self.readdir = os.path.join(self.dir, "read")
        self.sessions_file = os.path.join(self.dir, "sessions.jsonl")

    # --- sessions that orchestrate trainworld -------------------------------------------------------
    def handoff_sessions(self):
        try:
            with open(os.path.join(self.root, "notes", "handoff.md"), encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            return {}
        return {m.group(1): m.group(2) for m in SESSION_LINE.finditer(text)}

    def registered(self):
        """{session id: registered at}; the handoff's Orchestrator and Previous sessions count too."""
        out = {}
        try:
            with open(self.sessions_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                        out.setdefault(r["session"], r.get("at"))
                    except (ValueError, KeyError):
                        continue
        except OSError:
            pass
        for sid in self.handoff_sessions().values():
            out.setdefault(sid, None)
        return out

    def register(self, sid):
        if self._in_file(sid):
            return False
        os.makedirs(self.dir, exist_ok=True)
        with open(self.sessions_file, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"session": sid, "at": now_iso()}) + "\n")
        return True

    def _in_file(self, sid):
        try:
            with open(self.sessions_file, encoding="utf-8") as f:
                return any(f'"{sid}"' in l for l in f)
        except OSError:
            return False

    def accepts(self, sid):
        return any(sid == s or (len(s) >= 8 and sid.startswith(s)) for s in self.registered())

    # --- report files ----------------------------------------------------------------------------------
    def exists(self, rid):
        return any(os.path.exists(os.path.join(d, rid + ".json")) for d in (self.dir, self.readdir))

    def write(self, rec):
        if self.exists(rec["id"]):
            return False
        os.makedirs(self.dir, exist_ok=True)
        tmp = os.path.join(self.dir, f".{rec['id']}.{os.getpid()}.{time.time_ns()}.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(self.dir, rec["id"] + ".json"))
        return True

    def add(self, name, text, task=None, kind="gemini", session="gemini", agent_id=None, model=None, description=""):
        """Keep a report from an agent outside Claude's hooks. The watch prints it as long as session is not the
        watching orchestrator's id, so "gemini" rather than a session id."""
        agent_id = agent_id or f"{kind}-{name}"
        rid = "r-" + hashlib.sha1(f"{agent_id}\0{kind}\0{text}".encode("utf-8")).hexdigest()[:8]
        rec = {"id": rid, "at": now_iso(), "session": session, "agent_id": agent_id, "name": name, "task": task,
               "description": description, "model": model, "kind": kind, "text": text}
        self.write(rec)
        return rec

    def load(self, d):
        out = []
        for p in glob.glob(os.path.join(d, "r-*.json")):
            try:
                with open(p, encoding="utf-8") as f:
                    out.append(json.load(f))
            except (OSError, ValueError):
                continue
        return sorted(out, key=lambda r: r.get("at") or "")

    def unread(self):
        return self.load(self.dir)

    def all(self):
        return self.load(self.dir) + self.load(self.readdir)

    def mark_read(self, rec, by, note=""):
        src = os.path.join(self.dir, rec["id"] + ".json")
        if not os.path.exists(src):
            return False
        os.makedirs(self.readdir, exist_ok=True)
        rec = dict(rec, read_by=by, read_at=now_iso(), read_note=note)
        tmp = os.path.join(self.readdir, f".{rec['id']}.{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(self.readdir, rec["id"] + ".json"))
        try:
            os.remove(src)
        except OSError:
            pass
        return True

    def read_task(self, task, by, note=""):
        """Mark every unread report for this task read; returns how many."""
        return sum(self.mark_read(r, by, note) for r in self.unread() if r.get("task") == task)

    def current(self):
        """The session id on the handoff's Orchestrator line, or ''."""
        return self.handoff_sessions().get("Orchestrator", "")

    def board_name(self, task):
        try:
            with open(os.path.join(self.root, "notes", "board-live.json"), encoding="utf-8") as f:
                agents = json.load(f).get("agents") or []
        except (OSError, ValueError):
            return None
        for a in agents:
            if task and task in str(a.get("task", "")):
                return a.get("name")
        return None


def match(rec, key):
    k = key.lower()
    return k in (str(rec.get("id", "")).lower(), str(rec.get("task", "")).lower(), str(rec.get("name", "")).lower(),
                 str(rec.get("agent_id", "")).lower())


# --- reading a subagent transcript -------------------------------------------------------------------------
def entries(path):
    out = []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue                     # a line still being written
                if isinstance(e, dict):
                    out.append(e)
    except OSError:
        pass
    return out


def is_prompt(e):
    """A turn the agent was given: its brief, or a message that resumed it (not a tool result)."""
    if e.get("type") != "user":
        return False
    c = (e.get("message") or {}).get("content")
    if isinstance(c, str):
        return bool(c.strip())
    return any(isinstance(x, dict) and x.get("type") == "text" for x in (c or []))


def first_prompt_text(es):
    for e in es:
        if is_prompt(e):
            c = e["message"]["content"]
            return c if isinstance(c, str) else " ".join(x.get("text", "") for x in c if isinstance(x, dict))
    return ""


def deliveries(seg):
    """What a transcript sent to its caller: successful SubagentHandback calls (a refused repeat is not one) and
    SendMessage to main, in order. Returns [(kind, timestamp, text, tool_use_id)] and the last assistant text."""
    results = {}
    for e in seg:
        if e.get("type") == "user":
            for x in (e.get("message") or {}).get("content") or []:
                if isinstance(x, dict) and x.get("type") == "tool_result":
                    results[x.get("tool_use_id")] = json.dumps(x.get("content"), ensure_ascii=False)
    out, last_text = [], ""
    for e in seg:
        if e.get("type") != "assistant":
            continue
        for x in (e.get("message") or {}).get("content") or []:
            if not isinstance(x, dict):
                continue
            if x.get("type") == "text" and x.get("text", "").strip():
                last_text = x["text"]
            if x.get("type") != "tool_use":
                continue
            inp = x.get("input") or {}
            res = results.get(x.get("id"), "")
            if x.get("name") == "SubagentHandback" and '\\"success\\":true' in res.replace(" ", ""):
                out.append(("handback", e.get("timestamp"), inp.get("message", ""), x.get("id")))
            elif x.get("name") == "SendMessage" and str(inp.get("to", "")).lower() == "main" and res:
                out.append(("message", e.get("timestamp"), inp.get("message", ""), x.get("id")))
    return out, last_text


def mentions_handback(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        return "delivered through SubagentHandback" in text or '"name":"SubagentHandback"' in text
    except OSError:
        return False


def meta_for(transcript):
    p = re.sub(r"\.jsonl$", ".meta.json", transcript)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record(box, session, agent_id, meta, es, kind, ts, text):
    desc = meta.get("description", "")
    task = (TASK.search(desc) or TASK.search(first_prompt_text(es)[:400]) or [None])[0]
    name = box.board_name(task)
    if not name:
        m = AGENT_NAME.search(first_prompt_text(es)[:2000])
        name = m.group(1) if m else None
    rid = "r-" + hashlib.sha1(f"{agent_id}\0{kind}\0{text}".encode("utf-8")).hexdigest()[:8]
    return {"id": rid, "at": ts or now_iso(), "session": session, "agent_id": agent_id, "name": name or desc[:40],
            "task": task, "description": desc, "model": meta.get("model"), "kind": kind, "text": text}


def is_compaction_summary(text):
    """The summary a compaction writes starts with an <analysis> or <summary> block (r-f11676a4)."""
    return re.match(r"\s*<(analysis|summary)>", text) is not None


def from_stop(box, data):
    """Records for one SubagentStop: what the agent's latest turn delivered."""
    sid = data.get("session_id") or ""
    path = data.get("agent_transcript_path") or ""
    agent_id = data.get("agent_id") or os.path.basename(path)[6:-6]
    if not sid or not box.accepts(sid):
        return []
    meta = meta_for(path)
    if int(meta.get("spawnDepth") or 1) > 1:
        return []                                # an agent's own helper reports to that agent, not to us
    es = entries(path)
    # Every delivery in the transcript: ids are content hashes, so ones already kept (or read) are skipped.
    sent, _ = deliveries(es)
    if not sent:
        # An agent told to report through SubagentHandback (the harness reminds it in the transcript) that stops
        # without one is pausing on its own background work, not reporting. Agents without the tool (headless
        # sessions, and interactive ones as late as 2026-11-16) report by their last text.
        if mentions_handback(path):
            return []
        text = (data.get("last_assistant_message") or "").strip()
        if not text or not meta or is_compaction_summary(text):
            return []        # no .meta.json: not a spawned agent (an orchestrator's own tab), nothing to report
        sent = [("final-text", None, text, None)]
    return [record(box, sid, agent_id, meta, es, k, ts, t) for k, ts, t, _ in sent]


REPORT_MAX_LINES = 18
REPORT_TARGET = 15


def too_long(box, data):
    """The block reason if this stop's final-text report is over the limit and the agent hasn't been blocked yet."""
    sid = data.get("session_id") or ""
    path = data.get("agent_transcript_path") or ""
    if data.get("stop_hook_active") or not sid or not box.accepts(sid):
        return None
    meta = meta_for(path)
    if not meta or int(meta.get("spawnDepth") or 1) > 1:
        return None
    text = (data.get("last_assistant_message") or "").strip()
    if not text or is_compaction_summary(text):
        return None
    if deliveries(entries(path))[0] or mentions_handback(path):
        return None                              # it reports through SubagentHandback: already sent, can't be held
    n = sum(1 for l in text.splitlines() if l.strip())
    if n <= REPORT_MAX_LINES:
        return None
    agent_id = data.get("agent_id") or os.path.basename(path)[6:-6]
    mark = os.path.join(box.dir, "blocked", re.sub(r"[^\w.-]", "_", agent_id) + ".txt")
    if os.path.exists(mark):
        return None
    os.makedirs(os.path.dirname(mark), exist_ok=True)
    with open(mark, "w", encoding="utf-8") as f:
        f.write(f"{now_iso()} {n} lines\n")
    return (f"Your report is {n} lines; the orchestrator's limit is {REPORT_TARGET}. Rewrite it in {REPORT_TARGET} lines "
            "or fewer: what changed and where, check results, strings, anything for Anita. Detail goes in your notes file.")


def projects_dir():
    return os.path.join(os.path.expanduser("~"), ".claude", "projects")


def sweep(box, sessions, since):
    made = []
    for sid in sessions:
        for path in glob.glob(os.path.join(projects_dir(), "*", sid, "subagents", "agent-*.jsonl")):
            meta = meta_for(path)
            if int(meta.get("spawnDepth") or 1) > 1:
                continue
            if since and os.path.getmtime(path) < since.timestamp():
                continue
            es = entries(path)
            agent_id = os.path.basename(path)[6:-6]
            for k, ts, t, _ in deliveries(es)[0]:
                d = parse_t(ts)
                if since and d and d < since:
                    continue
                rec = record(box, sid, agent_id, meta, es, k, ts, t)
                if box.write(rec):
                    made.append(rec)
    return made


def line(r, width=110):
    first = next((l.strip() for l in (r.get("text") or "").splitlines() if l.strip()), "")
    head = f"{r['id']} {r.get('name') or '?'} {r.get('task') or ''} ({r.get('kind')}, session {str(r.get('session'))[:8]}, {r.get('at', '')[5:16].replace('T', ' ')}Z)"
    room = max(20, width - len(head) - 2)
    return f"{head}: {first[:room]}{'...' if len(first) > room else ''}"


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(HERE))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hook")
    l = sub.add_parser("list")
    l.add_argument("--all", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("keys", nargs="+")
    r = sub.add_parser("read")
    r.add_argument("keys", nargs="*")
    r.add_argument("--session", help="every unread report from this session (id or its first 8 characters)")
    r.add_argument("--by", required=True)
    r.add_argument("--note", default="")
    w = sub.add_parser("sweep")
    w.add_argument("--session", action="append")
    w.add_argument("--since", help="ISO time; default each session's registration, else 12 hours ago")
    g = sub.add_parser("register")
    g.add_argument("session")
    d = sub.add_parser("add")
    d.add_argument("--name", required=True)
    d.add_argument("--text-file", required=True)
    d.add_argument("--task")
    d.add_argument("--kind", default="gemini")
    d.add_argument("--model")
    a = ap.parse_args()
    box = Box(a.root)

    if a.cmd == "hook":
        try:
            data = json.loads(sys.stdin.buffer.read().decode("utf-8", errors="replace") or "{}")
            reason = too_long(box, data)
            if reason:
                print(json.dumps({"decision": "block", "reason": reason}))
                return 0                         # the rewrite reaches us on the next stop
            for rec in from_stop(box, data):
                box.write(rec)
        except Exception as ex:                  # never block or disturb an agent's stop
            try:
                os.makedirs(box.dir, exist_ok=True)
                with open(os.path.join(box.dir, "hook-errors.log"), "a", encoding="utf-8") as f:
                    f.write(f"{now_iso()} {type(ex).__name__}: {ex}\n")
            except OSError:
                pass
        return 0
    if a.cmd == "register":
        print("registered" if box.register(a.session) else "already registered")
        return 0
    if a.cmd == "add":
        with open(a.text_file, encoding="utf-8") as f:
            rec = box.add(a.name, f.read().strip(), task=a.task, kind=a.kind, model=a.model)
        print("  " + line(rec))
        return 0
    if a.cmd == "list":
        rs = box.all() if a.all else box.unread()
        for rec in rs:
            print(("  " if "read_by" not in rec else "  (read) ") + line(rec))
        if not rs:
            print("  no unread agent reports" if not a.all else "  no agent reports")
        return 0
    if a.cmd in ("show", "read"):
        pool = box.all() if a.cmd == "show" else box.unread()
        hits = [rec for rec in pool if any(match(rec, k) for k in a.keys)
                or (getattr(a, "session", None) and str(rec.get("session", "")).startswith(a.session))]
        if not hits:
            what = ", ".join(a.keys) or f"session {getattr(a, 'session', '')}"
            print("no " + ("" if a.cmd == "show" else "unread ") + "report matches " + what)
            return 0 if a.cmd == "read" and not a.keys else 1
        for rec in hits:
            if a.cmd == "show":
                print(f"== {line(rec, 200).split(': ')[0]}" + (f"; read by {rec['read_by']}" if rec.get("read_by") else ""))
                print(rec.get("text", ""))
            else:
                box.mark_read(rec, a.by, a.note)
                print(f"read: {rec['id']} {rec.get('name')} {rec.get('task') or ''}")
        return 0
    if a.cmd == "sweep":
        reg = box.registered()
        sessions = a.session or list(reg)
        made = []
        for sid in sessions:
            since = parse_t(a.since) if a.since else (parse_t(reg.get(sid)) if reg.get(sid) else
                                                       datetime.now(timezone.utc) - timedelta(hours=12))
            made += sweep(box, [sid], since)
        for rec in made:
            print("  new " + line(rec))
        print(f"{len(made)} new report(s)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
