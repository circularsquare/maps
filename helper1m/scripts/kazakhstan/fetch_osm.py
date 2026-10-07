"""Download Kazakhstan's rayon boundaries (OSM admin_level 6) from Overpass.

Two steps, both cached under helper1m/data/kazakhstan/raw/:
  osm_admin_tags.json  every admin relation at levels 4-8 in Kazakhstan, tags only
  osm_adm6/<id>.json   each level-6 relation with member-way geometry (out geom)

One `out geom` query for the whole country timed out (504) on both public
endpoints in 2026-10, so geometry is fetched in batches of relation ids.
overpass.kumi.systems hung without answering at the same time, so it is not
in the list; the mail.ru mirror is the fallback.
prep_boundaries.py assembles polygons from these files.

Overpass refuses a default python-requests User-Agent (bare 406, or a 429 whose
body asks for a UA), so a real one is sent.
"""
import json
import sys
import time
from pathlib import Path

import requests

HELPER = Path(__file__).resolve().parents[2]
RAW = HELPER / "data" / "kazakhstan" / "raw"
TAGS = RAW / "osm_admin_tags.json"
GEOM_DIR = RAW / "osm_adm6"
UA = "helper1m-research/1.0"
ENDPOINTS = ["https://overpass-api.de/api/interpreter",
             "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
TAGS_QUERY = """[out:json][timeout:300];
area["ISO3166-1"="KZ"][admin_level=2]->.kz;
rel(area.kz)[boundary=administrative][admin_level~"^(4|5|6|7|8)$"];
out tags;"""
BATCH = 8


def overpass(q):
    for attempt in range(4):
        for ep in ENDPOINTS:
            try:
                r = requests.post(ep, data={"data": q}, headers={"User-Agent": UA}, timeout=300)
            except requests.RequestException as e:
                print(f"    {ep}: {e}")
                continue
            if r.status_code == 200:
                d = r.json()
                if d.get("elements"):
                    return d
                print(f"    {ep}: empty reply, remark {d.get('remark')}")
            else:
                print(f"    {ep}: HTTP {r.status_code}")
        time.sleep(20 * (attempt + 1))
    sys.exit("Overpass failed four rounds")


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    if not TAGS.exists():
        TAGS.write_text(json.dumps(overpass(TAGS_QUERY), ensure_ascii=False), encoding="utf-8")
    tags = json.loads(TAGS.read_text(encoding="utf-8"))
    ids = sorted(e["id"] for e in tags["elements"] if e["tags"].get("admin_level") == "6")
    # The three cities of republican significance (admin_level 4) too, to find
    # any district OSM lacks as the city minus the districts it has.
    ids += [e["id"] for e in tags["elements"] if e["tags"].get("admin_level") == "4"
            and e["tags"].get("name") in ("Астана", "Алматы", "Шымкент")]
    GEOM_DIR.mkdir(exist_ok=True)
    todo = [i for i in ids if not (GEOM_DIR / f"{i}.json").exists()]
    print(f"{len(ids)} relations, {len(todo)} to fetch")
    for k in range(0, len(todo), BATCH):
        chunk = todo[k:k + BATCH]
        q = f"[out:json][timeout:600];rel(id:{','.join(map(str, chunk))});out geom;"
        d = overpass(q)
        for e in d["elements"]:
            (GEOM_DIR / f"{e['id']}.json").write_text(json.dumps(e, ensure_ascii=False), encoding="utf-8")
        print(f"  {k + len(chunk)}/{len(todo)}", flush=True)
        time.sleep(2)


if __name__ == "__main__":
    main()
