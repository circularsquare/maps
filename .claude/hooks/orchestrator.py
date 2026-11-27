"""Hooks for orchestrator continuity: the live handoff file each project keeps in notes/handoff.md.

    session-start  prints this session's id and where each project's handoff stands
    stop           once at 200k, a look at what filled the context; then once per 100k tokens past 500k that it is
                   time for a fresh session
    agent-guard    stops a handed-over orchestrator from starting new agents; starting one also resets
                   the intake count below
    reply-nudge    (PostToolUse) once per message: if her latest message has gone REPLY_AFTER seconds
                   with no reply text from this orchestrator, add a reminder to the tool result;
                   also the intake reminder (post is the same command)
    code-guard     (PreToolUse on edits and shell) refuses an orchestrator's edits to code under
                   app/, sim/, pipeline/, checks/ of trainworld and isochrone: brief an agent instead.
                   Anything else (notes, docs, tasks, tools) stays open. Subagents are never affected
                   (they share the orchestrator's session_id but their payloads carry agent_id).
                   Fails open on any internal error.

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
    """The reminder text, or None."""
    if data.get("agent_id"):          # a subagent's tool call, not the orchestrator
        return None
    sid = data.get("session_id", "")
    found = waiting_message(data.get("transcript_path", ""))
    if not sid or not found:
        return None
    sent, key = found
    waited = time.time() - sent
    if waited < REPLY_AFTER:
        return None
    if not any(sid in orch for _, _, orch, _, _ in handoffs()):
        return None
    state = STATE_DIR / f"{sid}.nudge"
    try:
        if state.read_text() == str(key):
            return None
    except OSError:
        pass
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state.write_text(str(key))
    return (f"Anita's message has waited {int(waited // 60)} min with no reply: reply to her "
            "now (a short text message, even mid-task), then go on.")


INTAKE_LIMIT = 40_000    # chars of tool output an orchestrator takes in before the delegate reminder (Anita 2026-11-27)
INTAKE_GAP = 300         # seconds between two reminders at the earliest; and another INTAKE_LIMIT must come in
NOT_INTAKE = {"Agent", "SendMessage", "Poll", "Monitor", "TaskStop", "Skill", "ExitWorktree", "EnterWorktree"}


def _intake_size(data):
    """Characters this call put into the orchestrator's context, or None when it doesn't count. Calls made
    inside a REPL script (tool_use_id repl_...) are skipped: only what the script returns reaches the context,
    and that is the REPL call's own result."""
    name = data.get("tool_name", "")
    if name in NOT_INTAKE or str(data.get("tool_use_id", "")).startswith("repl_"):
        return None
    resp = data.get("tool_response")
    if name == "REPL" and isinstance(resp, dict):
        return sum(len(resp.get(k) or "") for k in ("result", "stdout", "stderr") if isinstance(resp.get(k), str))
    if isinstance(resp, str):
        return len(resp)
    return len(json.dumps(resp, ensure_ascii=False)) if resp is not None else 0


def _intake_state(sid):
    path = STATE_DIR / f"{sid}.intake.json"
    try:
        return path, json.loads(path.read_text())
    except (OSError, ValueError):
        return path, {}


def intake_reset(sid, prompt_id=""):
    path, st = _intake_state(sid)
    st.update({"chars": 0, "since": 0, "prompt": prompt_id or st.get("prompt", "")})
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st))


def intake(data):
    """Reminder text once an orchestrator has taken in INTAKE_LIMIT chars since its last agent start or since
    Anita last wrote (a new prompt_id), else None. Never blocks."""
    sid = data.get("session_id", "")
    if data.get("agent_id") or not sid:
        return None
    size = _intake_size(data)
    if size is None:
        return None
    if not any(sid in orch for _, _, orch, _, _ in handoffs()):
        return None
    path, st = _intake_state(sid)
    pid = data.get("prompt_id") or ""
    if pid and st.get("prompt") and pid != st["prompt"]:
        st.update({"chars": 0, "since": 0})
    if pid:
        st["prompt"] = pid
    st["chars"] = st.get("chars", 0) + size
    st["since"] = st.get("since", 0) + size
    now = time.time()
    msg = None
    if st["chars"] >= INTAKE_LIMIT and st["since"] >= INTAKE_LIMIT and now - st.get("at", 0) >= INTAKE_GAP:
        st["at"], st["since"] = now, 0
        msg = (f"You have read about {st['chars'] // 1000}k characters of tool output since you last briefed an "
               "agent or Anita wrote. Delegate, don't investigate: brief an agent with the question "
               "(docs/orchestrator.md, rule 2).")
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(st))
    return msg


def post(data):
    """PostToolUse: every reminder that is due, as one additionalContext."""
    parts = []
    for fn in (reply_nudge, intake):
        try:
            text = fn(data)
        except Exception:
            text = None
        if text:
            parts.append(text)
    if parts:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse",
                                                 "additionalContext": " ".join(parts)}}))


PROTECTED = re.compile(r"/(?:trainworld|isochrone)/(?:app|sim|pipeline|checks)/", re.I)
OPEN_SUFFIXES = (".md", ".txt")          # a check index or a README is bookkeeping, not code
ALLOW_MINUTES = 30
SHELL_WRITES = re.compile(r"\bsed\s+(?:-[a-z]*i|--in-place)|\bperl\s+-[a-z]*i|\btee\b|\bSet-Content\b|\bAdd-Content\b"
                          r"|\bOut-File\b|\bgit\s+apply\b|\bpatch\b|\bcp\b|\bmv\b|\bCopy-Item\b|\bMove-Item\b"
                          r"|\bRemove-Item\b|\brm\b", re.I)
PROTECTED_TOKEN = re.compile(r"(?:trainworld|isochrone)[/\\](?:app|sim|pipeline|checks)[/\\]", re.I)


def _is_protected(path):
    p = "/" + str(path).replace("\\", "/").lstrip("/")
    return bool(PROTECTED.search(p)) and not p.lower().endswith(OPEN_SUFFIXES)


def _shell_edits_code(cmd):
    """A shell command that writes into protected code: a redirect to it, or an editing/copying/removing
    command naming it (the file's .md/.txt names don't count)."""
    for m in re.finditer(r">>?\s*['\"]?([^\s'\"|;&]+)", cmd):
        if PROTECTED_TOKEN.search(m.group(1)) and _is_protected("/" + m.group(1).replace("\\", "/")):
            return True
    if not SHELL_WRITES.search(cmd):
        return False
    for tok in re.findall(r"[^\s'\"|;&<>]*(?:trainworld|isochrone)[/\\](?:app|sim|pipeline|checks)[/\\][^\s'\"|;&<>]*", cmd, re.I):
        if _is_protected("/" + tok.replace("\\", "/")):
            return True
    return False


def code_guard(data):
    """PreToolUse: refuse an orchestrator's own edits to code. Anything unexpected allows."""
    if data.get("agent_id"):
        return
    sid = data.get("session_id", "")
    name = data.get("tool_name", "")
    ti = data.get("tool_input") or {}
    if not sid or not isinstance(ti, dict):
        return
    if name in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        hit = _is_protected(ti.get("file_path") or ti.get("notebook_path") or "")
    elif name in ("Bash", "PowerShell"):
        cmd = ti.get("command") or ""
        hit = bool(PROTECTED_TOKEN.search(cmd)) and _shell_edits_code(cmd)
    else:
        return
    if not hit:
        return
    if not any(sid in orch for _, _, orch, _, _ in handoffs()):
        return
    try:
        if time.time() - (STATE_DIR / f"{sid}.allow-code").stat().st_mtime < ALLOW_MINUTES * 60:
            return
    except OSError:
        pass
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason":
            "An orchestrator doesn't edit code (Anita 11-27): brief an agent with what to change and where "
            "(docs/orchestrator.md, rule 2). Notes, tasks, docs, tools, the board and landing commands stay open. "
            f"For a real exception, such as a merge conflict while landing, create the file {STATE_DIR}/{sid}.allow-code "
            f"(good for {ALLOW_MINUTES} min), and say so to Anita.",
    }}))


def agent_guard(data):
    sid = data.get("session_id", "")
    desc = ((data.get("tool_input") or {}).get("description") or "").strip().lower()
    if not sid or desc.startswith("in-flight"):
        return
    if not data.get("agent_id"):          # the orchestrator briefing an agent: intake starts again from zero
        try:
            intake_reset(sid)
        except Exception:
            pass
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
    if not isinstance(data, dict):
        data = {}
    cmds = {"session-start": session_start, "stop": stop, "agent-guard": agent_guard,
            "reply-nudge": post, "post": post, "code-guard": code_guard}
    try:
        cmds[sys.argv[1]](data)
    except Exception:
        # A broken guard must never break a session: print nothing, which allows.
        if os.environ.get("ORCH_DEBUG"):
            raise


if __name__ == "__main__":
    main()
