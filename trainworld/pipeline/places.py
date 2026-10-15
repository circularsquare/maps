"""Each pack's place list from OpenStreetMap (T-177, T-065): towns, villages and neighbourhoods
inside the pack's footprint, for the search box and for naming a station placed while zoomed out.

    python pipeline/places.py              every registered city
    python pipeline/places.py nyc phl      some

Needs the pack's boundary (`data/packs/<id>.boundary.geojson`, from build_city.py). A city whose
config names a Geofabrik `extract` (tier 0) reads the extract's cached features
(osm_features); any other, one Overpass request over the boundary's bounding box, cached in
`data/raw/osm_places_<id>.json` (delete it to refetch; while it exists it is used for every
city). Points outside the footprint are dropped. Writes
`data/packs/<id>.places.json`, a separate file so the `.bin` pack is untouched:

    {"format": 1, "city": "nyc", "kinds": ["city", "town", ...],
     "places": [[name, kind index, lng, lat, population or 0, [other names]?], ...]}

Other names are OSM's alt_name, short_name, old_name and official_name, so "South Philly" can
find South Philadelphia when OSM records it. Data (c) OpenStreetMap contributors, ODbL.
"""
import datetime
import json
import os
import re
import sys
import time
import unicodedata

import requests
from shapely import contains_xy, make_valid
from shapely.geometry import shape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PACKS = os.path.join(ROOT, "data", "packs")
REGISTRY = os.path.join(ROOT, "app", "src", "game", "registry.json")
# Overpass answers a bare client with 406 or a false 429; a descriptive agent, no personal details.
UA = {"User-Agent": "trainworld-pipeline/0.1 (rail game data build; python-requests)"}
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter"]
STATUS = "https://overpass-api.de/api/status"

# Biggest first; the app reads the order from the file's "kinds".
KINDS = ["city", "town", "borough", "suburb", "quarter", "village", "neighbourhood", "hamlet"]
OTHER_NAMES = ["alt_name", "short_name", "old_name", "official_name", "name:en"]
DEDUP_M = 2000.0
LATIN_NAMES = ["name:en", "name:zh-Latn-pinyin", "name:zh_pinyin", "name:ja-Latn"]  # a city's "latin_names" (T-291)
KANA_NAMES = ["name:ja-Hira", "name:ja-Kana", "name:ja_kana"]  # readings for "kana_romaji" (T-420)


def query(bbox):
    s, w, n, e = bbox
    kinds = "|".join(KINDS)
    return (f'[out:json][timeout:300];nwr["place"~"^({kinds})$"]["name"]({s},{w},{n},{e});'
            "out center tags;")


def fetch(city, bbox):
    path = os.path.join(RAW, f"osm_places_{city}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    q = query(bbox)
    for attempt in range(6):
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=400)
        except requests.RequestException as ex:
            print(f"  {city}: {url} failed ({ex}); retrying", flush=True)
            time.sleep(10)
            continue
        if r.status_code == 200:
            data = r.json()
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, path)
            return data
        print(f"  {city}: {url} said {r.status_code}", flush=True)
        if r.status_code == 429:
            try:
                print("  status: " + requests.get(STATUS, headers=UA, timeout=30).text.replace("\n", " | "), flush=True)
            except requests.RequestException:
                pass
        time.sleep(30)
    raise SystemExit(f"{city}: Overpass failed")


def from_extract(extract, bbox):
    """The same elements from the Geofabrik extract's cached features (osm_features), an area at
    its bounding box centre as Overpass's `out center` gives it."""
    import osm_features
    s, w, n, e = bbox
    els = []
    for layer in ("points", "multipolygons"):
        df = osm_features.read(extract, layer, columns=["name", "place", "other_tags"], bbox=(w, s, e, n),
                               where=f"place IN {osm_features.PLACE_KINDS} AND name IS NOT NULL")
        for r in df.itertuples():
            tags = dict(re.findall(r'"([^"]*)"=>"([^"]*)"', r.other_tags or ""), place=r.place, name=r.name)
            if layer == "points":
                els.append({"type": "node", "lon": r.geometry.x, "lat": r.geometry.y, "tags": tags})
            else:
                x0, y0, x1, y1 = r.geometry.bounds
                els.append({"type": "area", "center": {"lon": (x0 + x1) / 2, "lat": (y0 + y1) / 2}, "tags": tags})
    return {"elements": els}


def norm(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^\w]+", " ", s).strip()


def build(city):
    with open(os.path.join(PACKS, f"{city}.boundary.geojson"), encoding="utf-8") as f:
        fc = json.load(f)
    foot = make_valid(shape(fc["features"][0]["geometry"]))
    w, s, e, n = foot.bounds
    bbox = (round(s, 4), round(w, 4), round(n, 4), round(e, 4))
    import build_city
    cfg = build_city.CITIES.get(city, {})
    extract = None
    if not os.path.exists(os.path.join(RAW, f"osm_places_{city}.json")):
        extract = cfg.get("extract")
    latin = cfg.get("latin_names", False)
    kana = __import__("japan").kana_romaji if cfg.get("kana_romaji") else None
    n_latin, name_src = 0, {}
    data = from_extract(extract, bbox) if extract else fetch(city, bbox)
    source = f"OpenStreetMap contributors, ODbL (Geofabrik {extract})" if extract else "OpenStreetMap contributors, ODbL (Overpass)"
    rows = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        kind = tags.get("place")
        name = (tags.get("name") or "").strip()
        if kind not in KINDS or not name:
            continue
        if el["type"] == "node":
            lng, lat = el["lon"], el["lat"]
        elif "center" in el:
            lng, lat = el["center"]["lon"], el["center"]["lat"]
        else:
            continue
        if not contains_xy(foot, lng, lat):
            continue
        pop = 0
        m = re.match(r"^\s*(\d[\d,]*)", tags.get("population", ""))
        if m:
            pop = int(m.group(1).replace(",", ""))
        others = []
        if latin:  # T-291: a Latin name to show, the local one kept for search
            src = next((k for k in LATIN_NAMES if (tags.get(k) or "").strip()), None)
            shown = tags[src].strip() if src else ""
            if not shown and kana:  # T-420: romaji from the kana reading
                src = next((k for k in KANA_NAMES if (tags.get(k) or "").strip()), None)
                shown = (kana(tags[src]) or "") if src else ""
                src = "kana" if shown else None
            name_src[src or "local"] = name_src.get(src or "local", 0) + 1
            if shown and shown != name:
                others.append(name)
                name = shown
                n_latin += 1
        for k in OTHER_NAMES:
            for v in (tags.get(k) or "").split(";"):
                v = v.strip()
                if v and norm(v) != norm(name) and v not in others:
                    others.append(v)
        rows.append({"name": name, "kind": kind, "lng": lng, "lat": lat, "pop": pop, "others": others,
                     "node": el["type"] == "node"})
    # One row per place: OSM often has a label node and a boundary for the same neighbourhood.
    # Same name within DEDUP_M: keep the node, then the bigger kind; merge other names, population.
    rows.sort(key=lambda r: (not r["node"], KINDS.index(r["kind"])))
    kept, by_name = [], {}
    for r in rows:
        k = norm(r["name"])
        twin = None
        for o in by_name.get(k, []):
            dx = (o["lng"] - r["lng"]) * 111320 * 0.76
            dy = (o["lat"] - r["lat"]) * 110540
            if (dx * dx + dy * dy) ** 0.5 < DEDUP_M:
                twin = o
                break
        if twin:
            twin["pop"] = twin["pop"] or r["pop"]
            twin["others"] += [v for v in r["others"] if v not in twin["others"]]
            continue
        by_name.setdefault(k, []).append(r)
        kept.append(r)
    kept.sort(key=lambda r: (KINDS.index(r["kind"]), -r["pop"], r["name"]))
    places = []
    for r in kept:
        row = [r["name"], KINDS.index(r["kind"]), round(r["lng"], 5), round(r["lat"], 5), r["pop"]]
        if r["others"]:
            row.append(r["others"])
        places.append(row)
    out = {"format": 1, "city": city, "source": source,
           "built": datetime.date.today().isoformat(), "kinds": KINDS, "places": places}
    path = os.path.join(PACKS, f"{city}.places.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    counts = {k: 0 for k in KINDS}
    for r in kept:
        counts[r["kind"]] += 1
    print(f"{city}: {len(places)} places ({len(rows)} before merging), {os.path.getsize(path) // 1024} KB: "
          + ", ".join(f"{k} {v}" for k, v in counts.items() if v), flush=True)
    if latin:
        shown = sum(1 for r in kept if all(ord(c) < 0x2E80 for c in r["name"]))
        print(f"{city}: Latin names on {n_latin} of {len(rows)} OSM places; {shown} of {len(kept)} kept places show a Latin name", flush=True)
        print(f"{city}: shown name by source: " + ", ".join(f"{k} {v} ({v / max(len(rows), 1):.1%})" for k, v in sorted(name_src.items(), key=lambda kv: -kv[1])), flush=True)


def main():
    with open(REGISTRY, encoding="utf-8") as f:
        ids = [c["id"] for c in json.load(f)["cities"]]
    for i, city in enumerate(sys.argv[1:] or ids):
        if i and not os.path.exists(os.path.join(RAW, f"osm_places_{city}.json")):
            time.sleep(5)  # be gentle with the public Overpass between cities
        build(city)


if __name__ == "__main__":
    main()
