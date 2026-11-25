"""Run a read-only Gemini (Antigravity CLI, agy) audit or review from Claude's orchestrator.

    venv\Scripts\python.exe trainworld/tools/gemini_audit.py "Audit ..."
    venv\Scripts\python.exe trainworld/tools/gemini_audit.py --brief trainworld/notes/audit-brief.md --out trainworld/research/audit.md
    venv\Scripts\python.exe trainworld/tools/gemini_audit.py --resume <conv-id> "Follow-up question..."

Gemini runs as the custom agent trainworld-reader: it reads files (anything under the maps root) and runs only commands
that read, git status, diff, log, show, blame, ls-files and grep, and Get-ChildItem (agy_session.read_rules). It can't
write files or start subagents. It works from --cwd (default: the current folder). A diff it should review goes in a file under artifacts/trainworld/gemini/, named in the brief.
Its reply has to end with a closing line; a turn that stops before it gets a nudge to carry on (at most 6). Until
T-4252 audits ran in plan mode with --sandbox: --sandbox does nothing on Windows 10, and a shell command Gemini tried
was refused by cancelling the whole turn, so T-4248 and T-4250 stopped after one line (agy_session.py). A run that
never reaches its closing line exits 1 and leaves --out alone. The conversation id goes to stderr, for --resume;
everything agy printed goes to artifacts/trainworld/gemini/logs/. Stdlib only.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agy_session  # noqa: E402

MAPS = HERE.parent.parent
LOGS = MAPS / "artifacts" / "trainworld" / "gemini" / "logs"
AGENT = "trainworld-reader"
TOOLS = ["view_file", "run_command", "command_status"]
BODY = """You are a careful reviewer reading Anita's maps repository for the trainworld orchestrator. You read files
with your file tool and search with git grep, git ls-files and Get-ChildItem, one command per call (no pipes or
chaining; any other command is refused and ends your turn). You change nothing. Answer from the code, with file paths
and line numbers, and say when you are unsure."""
MARK = "END OF ANSWER"


def minutes(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh]?)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError("a time like 30m, 1h or 900s")
    return float(m.group(1)) * {"s": 1 / 60, "m": 1, "h": 60, "": 1}[m.group(2)]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="Audit prompt text (or use --brief)")
    ap.add_argument("--brief", help="Path to a markdown/text brief file")
    ap.add_argument("--out", help="Write the answer to this file and print a short status line")
    ap.add_argument("--resume", help="Continue a previous agy conversation ID")
    ap.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--model", default="gemini-4-neon-medium", help="agy model (default Gemini 4 Neon medium, her 11-24)")
    ap.add_argument("--timeout", default="30m", type=minutes, help="whole run, e.g. 30m (default)")
    ap.add_argument("--cwd", help="Working directory (defaults to current directory)")
    ap.add_argument("--json", action="store_true", help="Print agy's final result event as JSON instead of the text")
    a = ap.parse_args()

    prompt = a.prompt or ""
    if a.brief:
        brief_text = Path(a.brief).read_text(encoding="utf-8")
        prompt = f"{brief_text}\n\n{prompt}".strip() if prompt else brief_text
    if not prompt:
        ap.error("provide a prompt or --brief <file>")
    prompt += f"\n\nWhen your answer is complete, end it with the line {MARK}."

    agy_session.ensure_agent(AGENT, "Read-only reviewer for trainworld/tools/gemini_audit.py", TOOLS, BODY)
    settings = {"agentMode": "", "toolPermission": "request-review",
                "permissions": {"allow": agy_session.read_rules() + ["read_file(%s)" % MAPS.as_posix()],
                                "deny": agy_session.backstop_rules()}}
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / ("audit-%s-%d.log" % (time.strftime("%Y%m%d-%H%M%S"), os.getpid()))
    sess = agy_session.Session(a.cwd or Path.cwd(), log, model=a.model, agent=AGENT, effort=a.effort,
                               settings=json.dumps(settings), conversation=a.resume)
    sess.start()
    try:
        res, nudges = agy_session.work(sess, prompt, MARK, time.time() + a.timeout * 60, max_nudges=6)
    finally:
        sess.close()
    sys.stderr.write("conversation %s, %d turns, log %s\n" % (sess.conversation, sess.turns, log))
    if res is None or MARK not in (res.get("response") or ""):
        why = sess.ended or "no closing line after %d nudges" % nudges
        sys.stderr.write("Gemini gave no complete answer (%s). Last words: %s\n"
                         % (why, (agy_session.reply_text(res, MARK) if res else sess.partial)[-800:]))
        sys.exit(1)
    text = agy_session.reply_text(res, MARK) + "\n"
    if a.json:
        print(json.dumps(res, ensure_ascii=False))
    elif a.out:
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:  # Python 3.9: write_text has no newline=
            f.write(text)
        print(f"wrote {out_path} ({len(text.splitlines())} lines)")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
