"""Write a saved fit's ends and PIs as lon,lat lines for the Rust edit test.

    python export_pis.py <name>
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parents[3] / "artifacts" / "trainworld" / "t727"  # outputs, not notes/
f = np.load(HERE / f"fit_{sys.argv[1]}.npz")
lon, lat, keep = f["lon"], f["lat"], f["keep"]
with open(HERE / f"pis_{sys.argv[1]}.txt", "w") as out:
    for i in keep:
        out.write(f"{lon[i]:.8f},{lat[i]:.8f}\n")
print(f"{keep.size} points")
