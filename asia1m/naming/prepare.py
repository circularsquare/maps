"""Refresh everything batches are made from, after more drawing in asia1m.ase.

    python asia1m/naming/prepare.py

Copies asia1m.ase (the original is never opened), exports its 'main' and 'text' layers, re-extracts
regions (../regionnaming/extract_regions.py) and rebuilds regions.json (facts.py). About a minute.
Admin rasters and pop.npy depend only on geography; rebuild those by hand when a new country is
drawn (../regionnaming/build_admin.py, pop_raster.py).
"""
import os, sys, shutil, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
RN = os.path.join(HERE, "..", "regionnaming")
ASE = os.path.join(HERE, "..", "asia1m.ase")
ASEPRITE = os.environ.get("ASEPRITE", r"C:/Program Files/Aseprite/Aseprite.exe")
copy = os.path.join(HERE, "_ase_copy.ase")
shutil.copyfile(ASE, copy)
for layer, out in [("main", os.path.join(RN, "main.png")), ("text", os.path.join(HERE, "text.png"))]:
    print(f"== exporting '{layer}' ==", flush=True)
    subprocess.run([ASEPRITE, "-b", "--layer", layer, copy, "--save-as", out], check=True)
os.remove(copy)
if os.path.getsize(os.path.join(RN, "main.png")) < 600_000:
    sys.exit("main.png looks empty; close any Aseprite window holding the file and re-run.")
env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
subprocess.run([sys.executable, os.path.join(RN, "extract_regions.py")], check=True, env=env)
subprocess.run([sys.executable, os.path.join(HERE, "facts.py"), os.path.join(HERE, "text.png")], check=True, env=env)
print("\nDone. Make a batch with: python asia1m/naming/batch.py <name> asia1m/naming/text.png --province <name>")
