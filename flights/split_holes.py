#!/usr/bin/env python3
"""split_holes.py — fan one combined paste file out into per-airport dumps.

Reads data/holes_pasted.txt, where each airport's flightsfrom.com page text
sits under a ">>> XXX" marker line (XXX = the airport's IATA, as listed in
data/holes_to_grab.txt). Writes each non-empty block to data/airport_<iata>.txt
— the exact files build_backfill.py consumes. Re-runnable; only blocks with real
content are written, so unfilled/skipped markers are ignored.

After running this: python build_backfill.py && python build_fr24.py
"""
import os, re, sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# default to holes_pasted.txt; pass a filename (under data/) for later rounds:
#   python split_holes.py round2_pasted.txt
SRC = os.path.join("data", sys.argv[1] if len(sys.argv) > 1 else "holes_pasted.txt")
MARK = re.compile(r"^>>>\s+([A-Za-z]{3})\b")

if not os.path.exists(SRC):
    raise SystemExit(f"missing {SRC} — paste the pages there first (see data/holes_to_grab.txt)")

lines = open(SRC, encoding="utf-8").read().splitlines()

blocks = {}          # iata -> list[str]
cur = None
for ln in lines:
    m = MARK.match(ln)
    if m:
        cur = m.group(1).upper()
        blocks.setdefault(cur, [])
        continue
    if cur is None:
        continue
    if ln.lstrip().startswith("###"):   # country header from the template -> not content
        continue
    blocks[cur].append(ln)

written, empty = [], []
for iata, body in blocks.items():
    # drop leading/trailing blank lines; treat a near-empty block as "not pasted"
    text = "\n".join(body).strip()
    if len(text) < 10:
        empty.append(iata)
        continue
    path = os.path.join("data", f"airport_{iata.lower()}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    written.append((iata, path, len(text)))

print(f"markers found : {len(blocks)}")
print(f"written       : {len(written)}")
for iata, path, n in sorted(written):
    print(f"   {iata}  -> {path}  ({n} chars)")
if empty:
    print(f"empty/skipped : {len(empty)}  {sorted(empty)}")
print("\nnext: python build_backfill.py && python build_fr24.py")
