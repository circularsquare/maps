"""Write run_<name>.cmd for wave 4: mkrun.py name ids jobs port "comment"
ids: a comma list, or @file holding one. --workers defaults to --jobs since T-966."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
name, ids, jobs, port, note = sys.argv[1:6]
if ids.startswith("@"):
    ids = open(os.path.join(HERE, ids[1:])).read().strip()
n = len(ids.split(","))
cmd = f"""@echo off
rem T-897 wave 4, run {name.upper()}: {note} ({n} rows), no join
cd /d C:\\Users\\anita\\projects\\maps
set PYTHONIOENCODING=utf-8
set TW_FAULT_S=900
set TW_SIM_DIR={HERE}\\simroot\\sim
set TW_CARGO_TARGET_DIR={HERE}\\simroot\\sim\\target
set PATH=%USERPROFILE%\\.cargo\\bin;%PATH%
venv\\Scripts\\python.exe -I trainworld\\pipeline\\world_run.py run --rows {ids} --no-join --jobs {jobs} --port {port} --coarse 1000 > C:\\Users\\anita\\projects\\maps\\artifacts\\trainworld\\t897-w4-{name}.log 2>&1
"""
open(os.path.join(HERE, f"run_{name}.cmd"), "w", newline="\r\n").write(cmd)
print(f"run_{name}.cmd", n, "rows")
