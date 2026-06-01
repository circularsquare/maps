"""Fetch OSM track centreline geometry for all London rail lines via Overpass.

Saves the raw Overpass response to data/api_cache/osm_routes.json. The response
contains route relations (one per line per direction/branch), the member ways
with actual rail centrelines, and the nodes those ways reference.

Run once, then re-run only when you need fresh OSM data.
"""
import json
import os
import urllib.parse
import urllib.request

OUT_PATH = "data/api_cache/osm_routes.json"
ENDPOINT = "https://overpass-api.de/api/interpreter"

QUERY = """
[out:json][timeout:300];
(
  relation["route"="subway"]["network"="London Underground"];
  relation["route"="light_rail"]["network"="Docklands Light Railway"];
  relation["route"="train"]
    ["network"~"^(London Overground|National Rail)$"]
    ["ref"~"^(Mildmay|Lioness|Windrush|Suffragette|Liberty|Weaver|Elizabeth)$"];
);
out body;
>;
out skel qt;
"""


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    print(f"POST {ENDPOINT}")
    print("  (Overpass typically takes 30-90s for this query)")
    body = urllib.parse.urlencode({"data": QUERY}).encode()
    req = urllib.request.Request(
        ENDPOINT,
        data=body,
        headers={"User-Agent": "londonriders-track-shapes/1.0"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read()
    print(f"  received {len(raw) / 1024 / 1024:.2f} MB")
    obj = json.loads(raw)
    counts = {}
    for e in obj.get("elements", []):
        counts[e["type"]] = counts.get(e["type"], 0) + 1
    print(f"  elements: {counts}")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"  wrote {OUT_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
