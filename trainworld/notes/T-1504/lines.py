"""T-1504: the save's lines from start.json (the probe's first dump): each line's stops placed at
the nearest CN registry zone, its length, its trains, and its weekday riders by purpose."""
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[2]
art = root.parent / "artifacts/trainworld/T-1504"
s = json.loads((art / "start.json").read_text(encoding="utf-8"))
reg = json.loads((root / "app/src/game/registry.json").read_text(encoding="utf-8"))
cn = [c for c in reg["cities"] if c.get("country") == "CN"]


def hav(a, b):
    r = math.pi / 180
    h = math.sin((b[1] - a[1]) * r / 2) ** 2 + math.cos(a[1] * r) * math.cos(b[1] * r) * math.sin((b[0] - a[0]) * r / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def near(p):
    z = min(cn, key=lambda c: hav(p, c["center"]))
    return z["id"], hav(p, z["center"])


print("line keys:", s["lineKeys"])
print("view keys:", s["view"]["keys"])
print("view numbers:", s["view"]["plain"])
pur = s["view"]["purposes"]
print("riders per day by purpose (sum over lines):", {k: v["sum"] for k, v in pur.items()})
for l in s["lines"]:
    st = [p for p in l["stops"] if p]
    length = sum(hav(a, b) for a, b in zip(st, st[1:]))
    ids = [near(p)[0] for p in st]
    by = {k: v["per"].get(str(l["id"]), 0) for k, v in pur.items()}
    extra = {k: v for k, v in l.items() if k not in ("stops", "id", "name", "colour", "color")}
    print(f"{l['id']:>4} {l.get('name', '')[:14]:14} {len(st):3} stops {length:6.0f} km  {extra}  riders {by}")
    if length > 150:
        print("      ", " ".join(ids))
