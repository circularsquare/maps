"""Download Russia's municipal boundaries (OSM admin_level=6) through Overpass.

Writes raw JSON under helper1m/data/russia/raw/osm/:
  tags.json            every boundary=administrative relation at admin_level 4
                       and 6 inside Russia's OSM area, tags only
  tags_city.json       admin_level 8 inside Moscow and St Petersburg (their
                       intra-city municipalities), tags only
  batch_NNN.json       relations (with member lists) + their member ways with
                       geometry, in batches of relation ids
  extra.json           Moscow / St Petersburg admin_level 5 (their okrugs and
                       districts) and the Russian-tagged Crimea and
                       Sevastopol relations
  extra_ids.json       single relations the area queries miss (Zelenograd)

The admin_level 8 relations of the two cities are fetched too (the batches
were cut before level 5 was chosen for them); prep_boundaries.py ignores them.

prep_boundaries.py assembles the polygons from these. Re-running skips batches
already on disk, so an interrupted run just continues.

Overpass needs a User-Agent: without one it answers 406, or a 429 that is not
really a rate limit. A busy server answers with an HTML page saying
"Dispatcher_Client::request_read_and_idx::timeout"; that is retried.
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

HELPER = Path(__file__).resolve().parents[2]
OUT = HELPER / "data" / "russia" / "raw" / "osm"
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://overpass.kumi.systems/api/interpreter"]
UA = "helper1m-research/1.0"
RU_AREA = 3600060189          # OSM relation 60189, Russian Federation
MOSCOW, SPB = 102269, 337422  # admin_level=4 relations of the two federal cities
# Russian-tagged admin_level=4 relations for Crimea and Sevastopol (OSM also
# carries Ukrainian ones, 72639 and 1574364).
CRIMEA, SEVASTOPOL = 3795586, 3788485
EXTRA_IDS = [1320358]  # Zelenogradsky administrative okrug, Moscow
BATCH = 40


def overpass(query, tries=8):
    data = query.encode("utf-8")
    for attempt in range(tries):
        url = ENDPOINTS[attempt % len(ENDPOINTS)]
        req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=1000) as r:
                body = r.read()
            if body.lstrip().startswith(b"{"):
                js = json.loads(body)
                if "remark" in js and "error" in js["remark"].lower():
                    raise RuntimeError(js["remark"][:200])
                return js
            raise RuntimeError(body[:300].decode("utf-8", "replace"))
        except Exception as e:  # busy server, timeout, 429 - wait and retry
            wait = 30 * (attempt + 1)
            print(f"  attempt {attempt + 1} on {url} failed: {str(e)[:160]}; waiting {wait}s",
                  flush=True)
            time.sleep(wait)
    raise SystemExit("Overpass kept failing")


def save(path, js):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(js, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tags_path = OUT / "tags.json"
    if not tags_path.exists():
        print("fetching admin_level 4/6 tags", flush=True)
        save(tags_path, overpass(
            f'[out:json][timeout:900];area({RU_AREA})->.ru;'
            'rel(area.ru)[boundary=administrative][admin_level~"^(4|6)$"];out tags;'))
    city_path = OUT / "tags_city.json"
    if not city_path.exists():
        print("fetching Moscow / St Petersburg admin_level 8 tags", flush=True)
        save(city_path, overpass(
            f'[out:json][timeout:900];'
            f'(area({3600000000 + MOSCOW});area({3600000000 + SPB});)->.c;'
            'rel(area.c)[boundary=administrative][admin_level=8];out tags;'))

    rels = json.loads(tags_path.read_text(encoding="utf-8"))["elements"]
    city = json.loads(city_path.read_text(encoding="utf-8"))["elements"]
    ids = sorted({e["id"] for e in rels if e["tags"].get("admin_level") == "6"}
                 | {e["id"] for e in city})
    print(f"{len(ids)} relations to fetch in batches of {BATCH}", flush=True)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    batches = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]
    for n, chunk in enumerate(batches[:limit]):
        path = OUT / f"batch_{n:03d}.json"
        if path.exists():
            continue
        t0 = time.time()
        js = overpass(
            "[out:json][timeout:900];"
            f"rel(id:{','.join(map(str, chunk))})->.r;"
            ".r out body;way(r.r);out skel geom;")
        save(path, js)
        print(f"batch {n + 1}/{len(batches)}: {len(js['elements'])} elements, "
              f"{path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s", flush=True)
        time.sleep(3)

    # Extras: Moscow's administrative okrugs and St Petersburg's districts
    # (admin_level 5), which are what level 2 uses inside the two cities, and
    # the Russian-tagged Crimea and Sevastopol relations, for INCLUDE_CRIMEA.
    extra_path = OUT / "extra.json"
    if (limit is None or limit >= len(batches)) and not extra_path.exists():
        print("fetching extras", flush=True)
        save(extra_path, overpass(
            "[out:json][timeout:900];"
            f"(area({3600000000 + MOSCOW});area({3600000000 + SPB});)->.c;"
            "(rel(area.c)[boundary=administrative][admin_level=5];"
            f"rel(id:{CRIMEA},{SEVASTOPOL});)->.r;"
            ".r out body;way(r.r);out skel geom;"))
    # Single relations the area queries miss: Moscow's Zelenograd okrug is an
    # exclave whose relation does not come back from area(Moscow).
    ids_path = OUT / "extra_ids.json"
    if (limit is None or limit >= len(batches)) and not ids_path.exists():
        print("fetching single relations", flush=True)
        save(ids_path, overpass(
            "[out:json][timeout:900];"
            f"rel(id:{','.join(map(str, EXTRA_IDS))})->.r;"
            ".r out body;way(r.r);out skel geom;"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
