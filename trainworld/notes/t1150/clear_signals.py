"""T-1150 rebuild: move each row's tier 0 signal cache (data/work/<id>.tier0_signals.npz) into
data/work/pre-T1150/signals/, so its next build reads GHSL afresh (bounded, townships) and the old
cache stays as the backup. The outline and OSM caches stay: the outlines do not change.
    python trainworld/notes/t1150/clear_signals.py id,id,...   (--dry to print only)"""
import os
import sys

TW = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
work = os.path.join(TW, "data", "work")
dest = os.path.join(work, "pre-T1150", "signals")
os.makedirs(dest, exist_ok=True)
n = 0
for i in sys.argv[1].split(","):
    p = os.path.join(work, f"{i}.tier0_signals.npz")
    if not os.path.exists(p):
        print(f"{i}: no cache")
        continue
    if "--dry" not in sys.argv:
        os.replace(p, os.path.join(dest, os.path.basename(p)))
    n += 1
print(f"{n} caches {'would move' if '--dry' in sys.argv else 'moved'} to {dest}")
