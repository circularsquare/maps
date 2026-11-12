"""How many zones show one name on the map and another ("X region") in the long-trip lists."""
import json, re
G = r"C:\Users\anita\projects\maps\trainworld\app\src\game"
reg = json.load(open(G + r"\registry.json", encoding="utf-8"))
hand = json.load(open(G + r"\zone-names.json", encoding="utf-8"))
places = reg["market"]["places"]


def clean(raw):
    s = re.sub(r"\s*\[[^\]]*\]", "", raw)
    s = re.sub(r"\s*\([^)]*\)\s*$", "", s)
    s = re.sub(r" CC\b", "", s)
    return re.sub(r"\s+", " ", s).strip() or raw


n = diff = 0
ex = []
for k, v in reg.items():
    if not isinstance(v, list):
        continue
    for z in v:
        if not (isinstance(z, dict) and "place" in z and "id" in z) or z["place"] < 0:
            continue
        n += 1
        shown = hand.get(z["id"], {}).get("name") or clean(z["name"])
        lst = places[z["place"]]["regionName"]
        if lst != shown + " region":
            diff += 1
            if len(ex) < 25:
                ex.append(f"{z['id']}: map '{shown}', lists '{lst}'")
print(f"{diff} of {n} zones differ")
print("\n".join(ex))
