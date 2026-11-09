"""Writes countries.json beside this file: start zone code -> ISO country code, from the game's
registry. Rerun when cities are added so new start codes group under their country."""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
reg = json.load(open(os.path.join(here, "..", "..", "app", "src", "game", "registry.json"), encoding="utf-8"))
out = {c["id"]: c["country"] for c in reg["cities"] if c.get("country")}
with open(os.path.join(here, "countries.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"), sort_keys=True)
print(len(out), "codes")
