"""Start the next orchestrator in a new Windows Terminal tab (Anita 2026-11-16: orchestrators restart themselves
at ~400k tokens; she works in Windows Terminal).

    venv\Scripts\python.exe trainworld/tools/ops/next_orchestrator.py <project> [--model opus] [--advisor fable] [--dry-run]

Opens a tab in her most recent Terminal window, in the maps root, running `claude --model <model> "/takeover <project>"`.
Run it only after notes/handoff.md is current; the new session claims the file and messages the old one.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

MAPS = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--advisor", help="advisor model for that session only, e.g. fable (Anita 2026-11-16 trial)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not (MAPS / a.project / "notes" / "handoff.md").exists():
        raise SystemExit(f"no {a.project}/notes/handoff.md under {MAPS}")
    wt = shutil.which("wt") or shutil.which("wt.exe")
    if not wt:
        raise SystemExit("Windows Terminal (wt.exe) not found")
    cmd = [wt, "-w", "0", "new-tab", "--title", f"{a.project} orchestrator", "-d", str(MAPS),
           "claude", "--model", a.model] + (["--advisor", a.advisor] if a.advisor else []) + [f"/takeover {a.project}"]
    print(" ".join(cmd))
    if not a.dry_run:
        # wt hands its caller's environment to the new tab; launched from inside a Claude session that carries
        # CLAUDECODE and friends, the new session counts itself a child and never registers as a messageable peer.
        env = {k: v for k, v in os.environ.items()
               if not (k.upper().startswith("CLAUDE") or k.upper() == "AI_AGENT")}
        env["CLAUDE_CODE_FORCE_SESSION_PERSISTENCE"] = "1"  # Anita 2026-11-16: keep transcripts
        subprocess.Popen(cmd, cwd=str(MAPS), env=env)
        print("started")


if __name__ == "__main__":
    main()
