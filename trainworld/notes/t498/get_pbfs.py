"""Download Geofabrik extracts (names after download.geofabrik.de/) into a folder."""
import sys, os, requests
out = sys.argv[1]
UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}
for name in sys.argv[2:]:
    p = os.path.join(out, name.split("/")[-1] + ".osm.pbf")
    if os.path.exists(p):
        continue
    with requests.get(f"https://download.geofabrik.de/{name}-latest.osm.pbf", headers=UA, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(p + ".part", "wb") as f:
            for c in r.iter_content(1 << 20):
                f.write(c)
    os.replace(p + ".part", p)
    print(name, os.path.getsize(p) // 1_000_000, "MB", flush=True)
