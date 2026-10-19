"""Collect picked names from every batch's feedback.txt into picks.csv (key, name, batch, n).

    python asia1m/naming/picks.py

batch.py reads picks.csv so later batches see their neighbours' new names. Names already on the
map before this tool are not in it; agents read those off the pictures.
"""
import os, csv, json, re
HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "batches")
rows = []
for name in sorted(os.listdir(B)):
    fb = os.path.join(B, name, "feedback.txt")
    fj = os.path.join(B, name, "facts.json")
    if not (os.path.exists(fb) and os.path.exists(fj)): continue
    key_of = {r["n"]: r["key"] for r in json.load(open(fj, encoding="utf-8"))["regions"]}
    for line in open(fb, encoding="utf-8"):
        m = re.match(r"\s*(\d+):\s*(.*)", line)
        if not m: continue
        p = re.search(r"\bpick ([^.]+)", m.group(2))
        if p and int(m.group(1)) in key_of:
            rows.append(dict(key=key_of[int(m.group(1))], name=p.group(1).strip(), batch=name, n=int(m.group(1))))
# regions named outside any batch (e.g. ones she split in the .ase later)
mp = os.path.join(HERE, "manual_picks.csv")
if os.path.exists(mp):
    for m in csv.DictReader(open(mp, encoding="utf-8")):
        rows.append(dict(key=m["key"], name=m["name"], batch="manual", n=0))
with open(os.path.join(HERE, "picks.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["key", "name", "batch", "n"])
    w.writeheader(); w.writerows(rows)
json.dump({r["key"]: r["name"] for r in rows}, open(os.path.join(B, "picks.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=0)   # for the viewer
print(f"{len(rows)} picks -> picks.csv, batches/picks.json")
