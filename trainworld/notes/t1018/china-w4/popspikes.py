"""Densest residents cell of each pack (ids, comma list), sorted: popspikes.py ids"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
TW = r"C:\Users\anita\projects\maps\trainworld"
sys.path.insert(0, os.path.join(TW, "pipeline"))
sys.path.insert(0, os.path.join(TW, "checks"))
import numpy as np
import read_pack, pack_gate
out = []
for cid in sys.argv[1].split(","):
    try:
        header, a = read_pack.load_pack(cid, pack_gate.PACKS)
    except Exception as e:
        continue
    i = int(np.argmax(a["pop"]))
    out.append((float(a["pop"][i]), cid, [round(float(v), 4) for v in pack_gate.lonlat(header, a["x_m"][i], a["y_m"][i])]))
out.sort(reverse=True)
for v in out[:15]:
    print(f"{v[1]} {v[0]:,.0f} at {v[2]}")
print("n", len(out), "median", f"{sorted(v[0] for v in out)[len(out) // 2]:,.0f}")
