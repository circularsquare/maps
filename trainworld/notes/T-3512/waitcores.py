"""Wait until the core ledger has at least N free cores; checks every 5 minutes. Prints the list when it ends."""
import re
import subprocess
import sys
import time

N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
MAPS = r"C:\Users\anita\projects\maps"
PY = MAPS + r"\venv\Scripts\python.exe"
while True:
    out = subprocess.run([PY, MAPS + r"\cores.py", "list"], capture_output=True, text=True).stdout
    m = re.search(r"free (\d+)", out)
    free = int(m.group(1)) if m else 0
    # my own claim (settle) is mine to use
    mine = re.search(r"^settle\s+(\d+) core", out, re.M)
    if mine:
        free += int(mine.group(1))
    print(time.strftime("%H:%M:%S"), "free", free, flush=True)
    if free >= N:
        print(out, flush=True)
        break
    time.sleep(300)
