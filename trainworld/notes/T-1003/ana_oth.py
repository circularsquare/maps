"""T-1003: the "other way" variants: model walk + (own formula variant - own formula base); A/B/T."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ana import load, CN, OTHER, reg, tot
names = sys.argv[1].split(",")
print("| Zone | base2 own check | " + " | ".join(names) + " |")
print("|---" * (len(names) + 2) + "|")
for c in CN + OTHER:
    b = tot(load(c, 'base2')[1])
    cells = [f"{100*b[1]/b[0]:.0f} vs {100*b[18]/b[0]:.0f}"]
    for v in names:
        t = tot(load(c, v)[1])
        w = (b[1] + t[18] - b[18]) / b[0]
        A = t[9] / (t[9] + t[11]); B = t[12] / (t[12] + t[14]); T = t[15] / (t[15] + t[17])
        cells.append(f"{100*w:.0f} ({100*A:.0f}/{100*B:.0f}/{100*T:.0f})")
    print(f"| {reg[c]['name']} | " + " | ".join(cells) + " |")
