"""Writes countries.json and cities.json beside this file, from the game's registry:
countries.json is start zone code -> ISO country code; cities.json is start zone code ->
[lon, lat, name, country] for the stats page's map. Rerun when cities are added."""
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
reg = json.load(open(os.path.join(here, "..", "..", "app", "src", "game", "registry.json"), encoding="utf-8"))
out = {c["id"]: c["country"] for c in reg["cities"] if c.get("country")}
with open(os.path.join(here, "countries.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, separators=(",", ":"), sort_keys=True)
cities = {c["id"]: [round(c["origin"]["lon"], 3), round(c["origin"]["lat"], 3), c["name"], c.get("country", "")]
          for c in reg["cities"] if c.get("origin")}
with open(os.path.join(here, "cities.json"), "w", encoding="utf-8") as f:
    json.dump(cities, f, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
print(len(out), "codes,", len(cities), "cities")
