"""Names suggested for more than one region, or already picked elsewhere -> batches/dupes.json

    python asia1m/naming/dupes.py

A clash means the same Chinese characters (admin suffixes 市/县/区... ignored), not the same
spelling: Anita is fine with two places that only share a pinyin (项城 and 襄城 are both
Xiangcheng). Candidates without characters fall back to their spelling. The viewer looks clashes
up by candidate (batch, region, name) and shows them on the card.
"""
import os, sys, re, json, csv, glob
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
STEM = re.compile(r"(自治县|自治旗|林区|特区|新区|县|区|市|旗|地区|盟|自治州|街道|镇)$")
def ident(c):
    loc = (c.get("local") or "").strip()
    if not loc: return ("en", c["name"].lower())
    s = STEM.sub("", loc)
    return ("zh", s if len(s) >= 2 else loc)   # 新县 stays 新县, not 新; 乡 is kept (新乡)

cands = {}                              # (batch, n, name lower) -> identity
seen = defaultdict(set)                 # identity -> {"batch n"}
for f in sorted(glob.glob(f"{HERE}/batches/*/candidates.json")):
    c = json.load(open(f, encoding="utf-8"))
    b = c.get("batch") or os.path.basename(os.path.dirname(f))
    for r in c["regions"]:
        for k in r["candidates"]:
            idt = ident(k)
            cands[(b, r["n"], k["name"].lower())] = idt
            if k.get("status") != "rejected": seen[idt].add(f'{b} {r["n"]}')
picked = {}
if os.path.exists(f"{HERE}/picks.csv"):
    for p in csv.DictReader(open(f"{HERE}/picks.csv", encoding="utf-8")):
        idt = cands.get((p["batch"], int(p["n"]), p["name"].lower()), ("en", p["name"].lower()))
        picked[idt] = f'{p["batch"]} {p["n"]}'

out = {}
for (b, n, nm), idt in cands.items():
    here = f"{b} {n}"
    others = sorted(seen[idt] - {here, picked.get(idt)})
    pk = picked.get(idt) if picked.get(idt) != here else None
    if others or pk:
        out[f"{here}|{nm}"] = dict(places=others, picked=pk)
        print(f"{here} {nm} ({idt[1]}): " + ", ".join(others) + (f"  [picked for {pk}]" if pk else ""))
json.dump(out, open(f"{HERE}/batches/dupes.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
