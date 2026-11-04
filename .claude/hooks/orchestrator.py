"""Hooks for orchestrator continuity: the live handoff file each project keeps in notes/handoff.md.

    session-start  prints this session's id and where each project's handoff stands
    stop           warns once per 100k tokens past 500k that it is time for a fresh session
    agent-guard    stops a handed-over orchestrator from starting new agents

The skills that use this are .claude/skills/takeover and .claude/skills/handoff.
"""
import json
import os
import re
import sys
from pathlib import Path

MAPS = Path(__file__).resolve().parents[2]
CYCLE_AT = 500_000
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


def stop(data):
    tokens = context_tokens(data.get("transcript_path", ""))
    if tokens < CYCLE_AT:
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
        "systemMessage": f"Context is at {k}k tokens. A good point to start a fresh session "
                         "(/takeover <project> there, if this one is orchestrating).",
        "hookSpecificOutput": {
            "hookEventName": "Stop",
            "additionalContext": f"Context is at {k}k tokens, past the ~500k cycling point. If you are "
                                 "orchestrating, bring notes/handoff.md current (/handoff) so Anita can start "
                                 "a fresh orchestrator whenever she likes. A guideline: if cycling now "
                                 "would be worse (mid-landing, say), finish that first.",
        },
    }))


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
    {"session-start": session_start, "stop": stop, "agent-guard": agent_guard}[sys.argv[1]](data)


if __name__ == "__main__":
    main()
