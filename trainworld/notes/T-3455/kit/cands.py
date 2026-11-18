"""Ids whose share of pack borders is >= 0.8 (the only ones the sea pass can lift to 0.9), comma-joined."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
B = json.load(open(os.path.join(HERE, "borders.json"), encoding="utf8"))
out = []
for i, b in B.items():
    if not b["nb"]:
        continue
    tl = max(b["nb"].values())
    if tl / sum(b["nb"].values()) >= 0.8:
        out.append(i)
open(os.path.join(HERE, "cands.txt"), "w").write(",".join(out))
print(len(out))
