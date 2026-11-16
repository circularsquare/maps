"""Restart the 8800 server (maps/serve.py): stop the running one, start a new one detached.

Anita, 2026-11-16: serve.py restarts are fine at any time (they only affect her pages).
Usage, from anywhere: venv\Scripts\python.exe trainworld/tools/ops/restart_serve.py
"""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

MAPS = Path(__file__).resolve().parents[3]
PY = r"C:\Python39\python.exe"


def serve_pids():
    ps = ("Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
          "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    rows = json.loads(out) if out else []
    if isinstance(rows, dict):
        rows = [rows]
    pids = []
    for r in rows:
        cmd = (r.get("CommandLine") or "").strip()
        # only a bare "python serve.py" (the maps root server), never other servers or tools
        if re.fullmatch(r'"?[^"\s]*python(?:\.exe)?"?\s+serve\.py', cmd, re.I):
            pids.append(int(r["ProcessId"]))
    return pids


def main():
    pids = serve_pids()
    for pid in pids:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    time.sleep(1)
    flags = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED_PROCESS | NEW_PROCESS_GROUP | CREATE_NO_WINDOW
    subprocess.Popen([PY, "serve.py"], cwd=str(MAPS), creationflags=flags,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    for _ in range(20):
        time.sleep(0.5)
        try:
            with urllib.request.urlopen("http://localhost:8800/", timeout=2) as r:
                print(f"stopped {pids or 'none'}; 8800 answers {r.status}")
                return 0
        except Exception:
            pass
    print(f"stopped {pids or 'none'}; 8800 did not answer within 10 s", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
