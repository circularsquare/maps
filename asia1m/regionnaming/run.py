"""Regenerate regions_all.csv from asia1m.ase.

Exports the 'main' layer, then runs: extract -> admin -> townships -> describe -> finalize.
Override the Aseprite path with the ASEPRITE env var if needed.
"""
import os, sys, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ASE = os.path.abspath(os.path.join(HERE, "..", "asia1m.ase"))
ASEPRITE = os.environ.get("ASEPRITE", r"C:/Program Files/Aseprite/Aseprite.exe")
env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

print("== exporting 'main' layer ==")
main_png = os.path.join(HERE, "main.png")
subprocess.run([ASEPRITE, "-b", "--layer", "main", ASE, "--save-as", main_png], check=True)
# Aseprite occasionally exports an empty frame if a GUI instance has the file open.
if os.path.getsize(main_png) < 600_000:
    sys.exit(f"ERROR: {main_png} looks empty ({os.path.getsize(main_png)} bytes). "
             "Close the Aseprite GUI (or any open copy of the .ase) and re-run.")

for step in ["extract_regions.py", "build_admin.py", "build_townships.py",
             "describe_full.py", "finalize.py"]:
    print(f"== {step} ==")
    subprocess.run([sys.executable, os.path.join(HERE, step)], check=True, env=env)

print("\nDone -> regions_all.csv + REPORT.md")
