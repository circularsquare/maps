"""Run a non-interactive Antigravity (Gemini) read-only audit from Claude's orchestrator.

    venv\\Scripts\\python.exe trainworld/tools/gemini_audit.py "Audit ..."
    venv\\Scripts\\python.exe trainworld/tools/gemini_audit.py --brief trainworld/notes/audit-brief.md --out trainworld/research/audit.md
    venv\\Scripts\\python.exe trainworld/tools/gemini_audit.py --resume <conv-id> "Follow-up question..."

Runs `agy -p` in `--mode plan --sandbox` (read-only, sandboxed) with a timeout so an audit cannot edit files
or hang indefinitely. Resolves `agy.exe` from PATH or %LOCALAPPDATA%\\agy\\bin (works even in shells started
before agy was installed). Stdlib only.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_agy():
    exe = shutil.which("agy")
    if exe:
        return exe
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cand = Path(local) / "agy" / "bin" / "agy.exe"
        if cand.exists():
            return str(cand)
    home_cand = Path.home() / ".local" / "bin" / "agy"
    if home_cand.exists():
        return str(home_cand)
    raise SystemExit("agy CLI not found; install with: irm https://antigravity.google/cli/install.ps1 | iex")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", help="Audit prompt text (or use --brief)")
    ap.add_argument("--brief", help="Path to a markdown/text brief file")
    ap.add_argument("--out", help="Write full response to this file and print a short status line")
    ap.add_argument("--resume", help="Continue a previous agy conversation ID")
    ap.add_argument("--effort", default="medium", choices=("low", "medium", "high", "xhigh", "max"))
    ap.add_argument("--model", default="gemini-4-neon-medium", help="agy model (default Gemini 4 Neon medium, her 11-24)")
    ap.add_argument("--timeout", default="15m", help="Print-mode timeout (default 15m)")
    ap.add_argument("--cwd", help="Working directory (defaults to current directory)")
    ap.add_argument("--json", action="store_true", help="Emit agy JSON output (includes conversation ID)")
    a = ap.parse_args()

    prompt = a.prompt or ""
    if a.brief:
        brief_text = Path(a.brief).read_text(encoding="utf-8")
        prompt = f"{brief_text}\n\n{prompt}".strip() if prompt else brief_text
    if not prompt:
        ap.error("provide a prompt or --brief <file>")

    cmd = [
        find_agy(),
        "-p", prompt,
        "--mode", "plan",
        "--sandbox",
        "--effort", a.effort,
        "--print-timeout", a.timeout,
        "--output-format", "json" if a.json else "text",
    ]
    if a.resume:
        cmd += ["--conversation", a.resume]
    if a.model:
        cmd += ["--model", a.model]

    kw = {"capture_output": True, "text": True, "encoding": "utf-8", "errors": "replace"}
    if a.cwd:
        kw["cwd"] = a.cwd
    if sys.platform == "win32":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW

    p = subprocess.run(cmd, **kw)
    if p.returncode != 0:
        if p.stderr:
            sys.stderr.write(p.stderr)
        if p.stdout:
            sys.stdout.write(p.stdout)
        sys.exit(p.returncode)

    if a.out:
        out_path = Path(a.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:  # Python 3.9: write_text has no newline=
            f.write(p.stdout)
        lines = len(p.stdout.splitlines())
        print(f"wrote {out_path} ({lines} lines)")
    else:
        sys.stdout.write(p.stdout)


if __name__ == "__main__":
    main()
