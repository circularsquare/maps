import json, pickle, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"C:\Users\anita\projects\maps\noritetsu")
cc = sys.argv[1] if len(sys.argv) > 1 else "us"
pat = sys.argv[2] if len(sys.argv) > 2 else "PATH"
D = Path(sys.argv[3]) / cc if len(sys.argv) > 3 else ROOT / "dist" / "data" / cc
lines = json.loads((D / "lines.json").read_text(encoding="utf-8"))["lines"]
st = json.loads((D / "stations.json").read_text(encoding="utf-8"))["stations"]
ways = json.loads((D / "ways.json").read_text(encoding="utf-8"))
sys.path.insert(0, str(ROOT))
import ownership
foot = ownership.read(D / "foot.json")
gid2 = {}
for l in lines:
    for s in l["sections"]:
        gid2[s[3]] = (l, s)
with open(ROOT / "data" / "proc" / cc / "rels.pkl", "rb") as f:
    rels = pickle.load(f)
sel = [l for l in lines if pat in l["name"]]
wl = ways["lines"]
for l in sel:
    print("LINE", l["id"], l["name"], l.get("ref"), l["km"], l.get("src"))
    mid = int(l["id"][1:])
    if l["id"].startswith("m") and mid in rels:
        for ty, r, role in rels[mid][1]:
            if ty == "r" and r in rels:
                t, m = rels[r]
                wids = [ref for ty2, ref, ro in m if ty2 == "w" and (not ro or ro.startswith(("forward", "backward")))]
                own = [wl[ways["ways"][str(w)][0]] if str(w) in ways["ways"] else None for w in wids]
                print("   rel", r, t.get("name"), "from", t.get("from"), "to", t.get("to"), len(wids), "ways")
                print("      ways:", wids[:40])
                print("      owner:", [o for o in own][:40])
    for s in l["sections"]:
        a, b, km, g = s[:4]
        f = foot.get(g)
        desc = []
        for e in f or []:
            ol, os_ = gid2[e[0]]
            desc.append(f"{ol['name'][:30]}:{st[os_[0]]['n']}-{st[os_[1]]['n']} [{e[1]:.2f}-{e[2]:.2f}] a{e[3]:.2f}-{e[4]:.2f}")
        print(f"   sec {g} {st[a]['n']} - {st[b]['n']} {km} -> {'SELF' if f is None else desc}")
