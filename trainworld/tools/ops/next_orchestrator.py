"""Start the next orchestrator in a new Windows Terminal tab (Anita 2026-11-16: orchestrators restart themselves
at ~500k tokens; she works in Windows Terminal).

    venv\Scripts\python.exe trainworld/tools/ops/next_orchestrator.py <project> [--model opus] [--advisor fable] [--account 1|2] [--dry-run]

Opens a tab in her most recent Terminal window, in the maps root, running `claude --model <model> "/takeover <project>"`.
Without --account the tab starts on the account of the session running this script (its CLAUDE_CONFIG_DIR; unset
is account 1). --account 1|2 overrides, e.g. to move to the other login when this one runs low (see claude2.py).
Run it only after notes/handoff.md is current; the new session claims the file and messages the old one.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

MAPS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAPS))
import claude2  # noqa: E402


def own_account():
    """(account number, why) for the session running this script, from CLAUDE_CONFIG_DIR as it was inherited."""
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    if not env:
        return 1, "CLAUDE_CONFIG_DIR is unset in this session"
    norm = lambda p: os.path.normcase(os.path.abspath(os.path.expanduser(p)))
    for i, d in enumerate(claude2.ACCOUNTS):
        if norm(env) == norm(d):
            return i + 1, f"this session runs with CLAUDE_CONFIG_DIR={env}"
    return 1, f"CLAUDE_CONFIG_DIR={env} matches no known account"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--advisor", help="advisor model for that session only, e.g. fable (Anita 2026-11-16 trial)")
    ap.add_argument("--standby", action="store_true",
                    help="open a standby tab that lists itself in <project>/notes/standby.txt and waits (Anita 2026-11-21)")
    ap.add_argument("--no-code-mode", action="store_true",
                    help="start without Code Mode (on by default, Anita 2026-11-23: 'that is usual procedure now')")
    ap.add_argument("--new-window", action="store_true",
                    help="open it in a new Terminal window instead of a tab in the latest one (her 11-24: a frozen window)")
    ap.add_argument("--account", type=int, choices=(1, 2),
                    help="which Claude login the new session runs on (1: ~/.claude; 2: ~/.claude-2); default: this session's")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.account:
        why = "--account given"
    else:
        a.account, why = own_account()
    print(f"account {a.account} ({why})")
    prompt = f"/takeover {a.project}"
    if a.standby:
        prompt = (f"You are a standby {a.project} orchestrator. Run ListAgents to learn your own name, append one line "
                  f"with your name and the date to {a.project}/notes/standby.txt, then wait and do nothing else "
                  f"until a message asks you to run /takeover {a.project}.")
    if not (MAPS / a.project / "notes" / "handoff.md").exists():
        raise SystemExit(f"no {a.project}/notes/handoff.md under {MAPS}")
    wt = shutil.which("wt") or shutil.which("wt.exe")
    if not wt:
        raise SystemExit("Windows Terminal (wt.exe) not found")
    title = f"{a.project} standby" if a.standby else f"{a.project} orchestrator"
    cmd = [wt, "-w", "new" if a.new_window else "0", "new-tab", "--title", title, "-d", str(MAPS),
           "claude", "--model", a.model] + (["--advisor", a.advisor] if a.advisor else []) + [prompt]
    print(" ".join(cmd) + (f"  [CLAUDE_CONFIG_DIR={claude2.ACCOUNTS[1]}]" if a.account == 2 else ""))
    if not a.dry_run:
        # wt hands its caller's environment to the new tab; launched from inside a Claude session that carries
        # CLAUDECODE and friends, the new session counts itself a child and never registers as a messageable peer.
        env = {k: v for k, v in os.environ.items()
               if not (k.upper().startswith("CLAUDE") or k.upper() == "AI_AGENT")}
        if a.account == 2:  # after the filter, which also drops CLAUDE_CONFIG_DIR: account 1 gets the default folder
            env["CLAUDE_CONFIG_DIR"] = claude2.ACCOUNTS[1]
        env["CLAUDE_CODE_FORCE_SESSION_PERSISTENCE"] = "1"  # Anita 2026-11-16: keep transcripts
        if not a.no_code_mode:  # the filter above drops it with the rest; it is read at startup, terminal tabs only
            env["CLAUDE_CODE_EXPERIMENTAL_CODE_MODE"] = "1"
        subprocess.Popen(cmd, cwd=str(MAPS), env=env)
        print("started")


if __name__ == "__main__":
    main()
