import glob
import json
import os

D = os.path.dirname(os.path.abspath(__file__))
rows = {}
for p in sorted(glob.glob(f"{D}/real/*.json")):
    c, v = os.path.basename(p)[:-5].split("_", 1)
    j = json.load(open(p))
    cm, ev = j["day"]["commute"], j["day"]["everyday"]
    rail = cm[1] + ev[1]
    tot = cm[0] + ev[0]
    walk = cm[2] + ev[2]
    rows[(c, v)] = (rail, 100 * rail / (tot - walk), 100 * cm[1] / (cm[0] - cm[2]), 100 * rail / tot, 100 * walk / tot)
print("city var rail/day rail%(rail+car) commute% rail%all walk%all")
for (c, v), r in rows.items():
    b = rows.get((c, "before"))
    ch = f"{100 * (r[0] / b[0] - 1):+.1f}%" if b else ""
    print(c, v, f"{r[0] / 1e6:.2f}M", ch, *(f"{x:.1f}" for x in r[1:]))
