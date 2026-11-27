"""Hooks for orchestrator continuity: the live handoff file each project keeps in notes/handoff.md.

    session-start  prints this session's id and where each project's handoff stands
    stop           once at 200k, a look at what filled the context; then once per 100k tokens past 500k that it is
                   time for a fresh session
    agent-guard    stops a handed-over orchestrator from starting new agents
    reply-nudge    (PostToolUse) once per message: if her latest message has gone REPLY_AFTER seconds
                   with no reply text from this orchestrator, add a reminder to the tool result

The skills that use this are .claude/skills/takeover and .claude/skills/handoff.
"""
import json
import os
import re
import sys
import time
from pathlib import Path

MAPS = Path(__file__).resolve().parents[2]
CYCLE_AT = 500_000   # Anita 2026-11-17: orchestrators start their successor at ~500k (was 400k)
EARLY_AT = 200_000   # Anita 2026-11-27: one early look at what fills the context, long before cycling
BAND = 100_000
STATE_DIR = Path(os.environ.get("TEMP", MAPS / ".claude")) / "claude-orchestrator"


def handoffs():
    """(project, path, orchestrator line, previous line, updated line) for each notes/handoff.md,
    in maps/ and in the projects beside it (transituniversalis lives outside maps)."""
    out = []
    for path in sorted(MAPS.glob("*/notes/handoff.md")) + sorted(MAPS.parent.glob("*/notes/handoff.md")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = {}
        for key in ("Orchestrator", "Previous", "Updated"):
            m = re.search(rf"^{key}:\s*(.*)$", text, re.M)
            lines[key] = m.group(1).strip() if m else ""
        out.append((path.parent.parent.name, path, lines["Orchestrator"], lines["Previous"], lines["Updated"]))
    return out


def session_start(data):
    sid = data.get("session_id", "")
    msg = [f"Session id: {sid}."]
    found = handoffs()
    if found:
        parts = []
        for project, _, orch, _, updated in found:
            who = "this session" if sid and sid in orch else (orch.split(",")[0] or "nobody")
            parts.append(f"{project} (orchestrator {who}; updated {updated or '?'})")
        msg.append("Orchestrator handoffs in <project>/notes/handoff.md: " + "; ".join(parts)
                   + ". To start orchestrating one: /takeover <project>.")
    if data.get("source") == "compact":
        mine = [p for p, _, orch, _, _ in found if sid and sid in orch]
        if mine:
            msg.append(f"This session was just compacted. Re-read {mine[0]}/notes/handoff.md before acting: "
                       "it is more reliable than the summary.")
    print(" ".join(msg))


def context_tokens(transcript):
    """Context size of the latest main-thread reply, read from the end of the transcript."""
    try:
        with open(transcript, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 2_000_000))
            tail = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return 0
    for line in reversed(tail):
        if '"usage"' not in line:
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("isSidechain") or entry.get("type") != "assistant":
            continue
        u = (entry.get("message") or {}).get("usage") or {}
        return sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                            "cache_read_input_tokens", "output_tokens"))
    return 0


def early(data, tokens):
    """Once a session, at EARLY_AT: how fast the context filled, and where an orchestrator's usually goes."""
    state = STATE_DIR / f"{data.get('session_id', 'unknown')}-early.txt"
    if state.exists():
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state.write_text("1")
    try:
        mins = int((time.time() - os.path.getctime(data.get("transcript_path", ""))) // 60)
        age = f" after {mins} min"
    except (OSError, ValueError, TypeError):
        age = ""
    print(json.dumps({"hookSpecificOutput": {"hookEventName": "Stop", "additionalContext":
        f"Context is at {tokens // 1000}k tokens{age}. If this session orchestrates, look at what filled it: "
        "docs/orchestrator.md \"Your own context\" (verdict lines only, short briefs, no investigating, "
        "bookkeeping once a round, short messages)."}}))


def stop(data):
    tokens = context_tokens(data.get("transcript_path", ""))
    if tokens < EARLY_AT:
        return
    if tokens < CYCLE_AT:
        early(data, tokens)
        return
    band = tokens // BAND
    state = STATE_DIR / f"{data.get('session_id', 'unknown')}.txt"
    try:
        warned = int(state.read_text())
    except (OSError, ValueError):
        warned = 0
    if band <= warned:
        return
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state.write_text(str(band))
    k = tokens // 1000
    print(json.dumps({
        "systemMessage": f"Context is at {k}k tokens. If this session orchestrates, it starts its successor now.",
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": f"Context is at {k}k tokens, past the ~500k cycling point. If you are "
                                 "orchestrating: finish anything mid-landing, bring notes/handoff.md current "
                                 "(/handoff), then start your successor as your project's orchestrator doc "
                                 "describes (trainworld: docs/orchestrator.md, \"Your own cycling\": standby "
                                 "session first, else next_orchestrator.py with its flags), and wind down per "
                                 "/handoff once it messages you (Anita 2026-11-16). After handing over, send her "
                                 "one line, 'talk to <new name> now', and nothing more.",
        },
    }))


REPLY_AFTER = 180   # Anita 2026-11-23: she should get a reply within a couple of minutes
NUDGE_TAIL = 3_000_000


def _ts(s):
    from datetime import datetime
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (ValueError, AttributeError):
        return 0.0


def _prompt_text(prompt):
    if isinstance(prompt, list):
        return "".join(b.get("text", "") for b in prompt if isinstance(b, dict))
    return prompt if isinstance(prompt, str) else ""


def waiting_message(transcript):
    """(time sent, key) of her latest message if no main-thread reply text follows it, else None.
    Reads the transcript backwards: the first reply or message found decides. A mid-turn message is a
    queued_command attachment (origin human); its enqueue entry holds the time she sent it."""
    try:
        with open(transcript, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - NUDGE_TAIL))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        human = '"kind":"human"' in line
        if not human and '"type":"assistant"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("type") == "assistant":
            if e.get("isSidechain"):
                continue
            content = (e.get("message") or {}).get("content")
            if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "text" and (b.get("text") or "").strip() for b in content):
                return None
            continue
        att = e.get("attachment")
        if e.get("type") == "user" and (e.get("origin") or {}).get("kind") == "human":
            return _ts(e.get("timestamp")), e.get("uuid") or e.get("timestamp")
        if e.get("type") == "attachment" and att and att.get("type") == "queued_command" \
                and (att.get("origin") or {}).get("kind") == "human":
            sent = _ts(e.get("timestamp"))
            want = _prompt_text(att.get("prompt")).strip()
            for j in range(i - 1, max(-1, i - 400), -1):
                if '"enqueue"' in lines[j] and want[:40] and want[:40] in lines[j]:
                    try:
                        q = json.loads(lines[j])
                    except ValueError:
                        continue
                    if q.get("operation") == "enqueue" and (q.get("content") or "").strip() == want:
                        sent = _ts(q.get("timestamp")) or sent
                        break
            return sent, e.get("uuid") or e.get("timestamp")
    return None


def reply_nudge(data):
    import time
    if data.get("agent_id"):          # a subagent's tool call, not the orchestrator
        return
    sid = data.get("session_id", "")
    found = waiting_message(data.get("transcript_path", ""))
    if not sid or not found:
        return
    sent, key = found
    waited = time.time() - sent
    if waited < REPLY_AFTER:
        return
    if not any(sid in orch for _, _, orch, _, _ in handoffs()):
        return
    state = STATE_DIR / f"{sid}.nudge"
    try:
        if state.read_text() == str(key):
            return
    except OSError:
        pass
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state.write_text(str(key))
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": f"Anita's message has waited {int(waited // 60)} min with no reply: reply to her "
                             "now (a short text message, even mid-task), then go on.",
    }}))


def agent_guard(data):
    sid = data.get("session_id", "")
    desc = ((data.get("tool_input") or {}).get("description") or "").strip().lower()
    if not sid or desc.startswith("in-flight"):
        return
    for project, path, orch, prev, _ in handoffs():
        if sid in prev and sid not in orch:
            print(json.dumps({"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason":
                    f"Another session has taken over orchestrating {project} ({orch}). Start no new work: "
                    "land what is in flight, update notes/handoff.md (/handoff), then stop. If this agent "
                    "is needed to finish in-flight work, start its description with 'in-flight:' and retry.",
            }}))
            return


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        data = json.loads(sys.stdin.buffer.read().decode("utf-8", errors="replace"))
    except ValueError:
        data = {}
    {"session-start": session_start, "stop": stop, "agent-guard": agent_guard,
     "reply-nudge": reply_nudge}[sys.argv[1]](data)


if __name__ == "__main__":
    main()
