"""T-2424: the place pairs whose routed drive takes a car ferry, with their trips' weight (the rows'
people), from a market-drives.bin.   python ferry_pairs.py BIN [N]"""
import json
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from zone_check import TW, read  # noqa: E402

ids, pairs, _ = read(sys.argv[1])
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
fe = [(ids[a], ids[b], v) for (a, b), v in pairs.items() if v[3] > 0]
print(f"{len(fe)} of {len(pairs)} pairs take a car ferry")
fe.sort(key=lambda t: -(rows[t[0]]["pop"] * rows[t[1]]["pop"]))
for a, b, v in fe[:int(sys.argv[2]) if len(sys.argv) > 2 else 40]:
    print(f"  {a} ({rows[a]['country']}) - {b} ({rows[b]['country']}): rt {v[0]:.2f} rk {v[1]:.2f}, {v[2]} km of ferry x{v[3]}")
