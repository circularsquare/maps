"""T-2424: the place pairs that took a car ferry in the old table, against the new: ferry or not, the ratios.
Also the named pairs.   python ferry_compare.py OLD.bin NEW.bin [a b ...]"""
import json
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from zone_check import TW, read  # noqa: E402

ids0, p0, _ = read(sys.argv[1])
ids1, p1, z1 = read(sys.argv[2])
rows = json.load(open(TW + r"\app\src\game\market-rows.json", encoding="utf-8"))
key1 = {(ids1[a], ids1[b]): v for (a, b), v in p1.items()}
get1 = lambda a, b: key1.get((a, b)) or key1.get((b, a))
old = [(ids0[a], ids0[b], v) for (a, b), v in p0.items() if v[3] > 0]
new_f = sum(1 for v in p1.values() if v[3] > 0)
print(f"old: {len(p0)} pairs, {len(old)} with a ferry; new: {len(p1)} pairs, {new_f} with a ferry, {len(z1)} zone detours")
still = [t for t in old if (get1(t[0], t[1]) or [0, 0, 0, 0])[3] > 0]
print(f"of the old ferry pairs, {len(still)} still take one (an OSM car ferry), {len(old) - len(still)} now drive round or swim")
w = lambda a, b: rows[a]["pop"] * rows[b]["pop"]
for a, b, v in sorted(old, key=lambda t: -w(t[0], t[1]))[:25]:
    n = get1(a, b)
    ns = f"rt {n[0]:.2f} rk {n[1]:.2f} ferry {n[2]} km x{n[3]}" if n else "no entry"
    print(f"  {a}-{b} ({rows[a]['country']}-{rows[b]['country']}): old rt {v[0]:.2f} rk {v[1]:.2f} ferry {v[2]} km; new {ns}")
names = sys.argv[3:]
for i in range(0, len(names) - 1, 2):
    a, b = names[i], names[i + 1]
    o = next((v for (x, y), v in p0.items() if {ids0[x], ids0[y]} == {a, b}), None)
    print(f"{a}-{b}: old {o}, new {get1(a, b)}")
