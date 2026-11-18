"""Each pack's place list from OpenStreetMap (T-177, T-065): towns, villages and neighbourhoods
inside the pack's footprint, for the search box and for naming a station placed while zoomed out.

    python pipeline/places.py              every registered city
    python pipeline/places.py nyc phl      some

Needs the pack's boundary (`data/packs/<id>.boundary.geojson`, from build_city.py). A city whose
config names a Geofabrik `extract` (tier 0) reads the extract's cached features
(osm_features); any other, one Overpass request over the boundary's bounding box, cached in
`data/raw/osm_places_<id>.json` (delete it to refetch; while it exists it is used for every
city). Points outside the footprint are dropped, but for a node in a narrow gap of it that no other
pack holds (`Gaps`, T-3623). Writes
`data/packs/<id>.places.json`, a separate file so the `.bin` pack is untouched:

    {"format": 1, "city": "nyc", "kinds": ["city", "town", ...],
     "places": [[name, kind index, lng, lat, population or 0, [other names]?], ...]}

Other names are OSM's alt_name, short_name, old_name and official_name, so "South Philly" can
find South Philadelphia when OSM records it. Data (c) OpenStreetMap contributors, ODbL.
"""
import datetime
import json
import math
import os
import re
import sys
import time
import unicodedata

import requests
from shapely import contains_xy, make_valid, prepare
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
DROP = object()  # a twin meaning "another pack lists this place"
LATIN_NAMES = ["name:en", "name:zh-Latn-pinyin", "name:zh_pinyin", "name:ja-Latn", "name:ko-Latn"]  # a city's "latin_names" (T-291)
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
                els.append({"type": "area", "center": {"lon": (x0 + x1) / 2, "lat": (y0 + y1) / 2}, "tags": tags,
                            "geom": r.geometry})
    return {"elements": els}


class Gaps:
    """T-3623: which place nodes just outside a pack's outline are still the pack's. Outlines have
    small holes and inlets: admin units leave a river out (Elephantine Island's villages at Aswan), or
    a small unit is in no pack (Cairo's node is in Qasr Al-Nile, 1.2 km2). A node in a gap of the
    outline narrower than about 2 x GAP_M (the outline closed by GAP_M) and in no other pack's outline
    is kept."""
    GAP_M = 500.0

    def __init__(self, city, foot):
        self.city, self.foot, self._closed, self._others = city, foot, None, None

    def _proj(self):
        import pyproj
        w, s, e, n = self.foot.bounds
        crs = pyproj.CRS.from_proj4(f"+proj=aeqd +lat_0={(s + n) / 2} +lon_0={(w + e) / 2} +units=m")
        return pyproj.Transformer.from_crs(4326, crs, always_xy=True).transform

    def holds(self, lng, lat):
        from shapely import dwithin
        from shapely.geometry import Point
        from shapely.ops import transform
        if not dwithin(self.foot, Point(lng, lat), self.GAP_M / (111320 * max(0.1, math.cos(math.radians(lat))))):
            return False
        if self._closed is None:
            self._fwd = self._proj()
            self._closed = transform(self._fwd, self.foot).buffer(self.GAP_M).buffer(-self.GAP_M)
            prepare(self._closed)
        if not self._closed.contains(Point(self._fwd(lng, lat))):
            return False
        return not self.elsewhere(lng, lat)

    def elsewhere(self, lng, lat):
        """True when another pack's outline holds the point: every boundary file in data/packs, as
        osm_units.cut_packs reads them, so packs built in the same run count before registering."""
        if self._others is None:
            import glob
            import osm_units
            paths = [p for p in glob.glob(os.path.join(PACKS, "*.boundary.geojson"))
                     if os.path.basename(p).split(".")[0] != self.city]
            self._others = {p: [box, None] for p, box in osm_units.pack_boxes(paths).items()}
        for p, o in self._others.items():
            w, s, e, n = o[0]
            if not (w <= lng <= e and s <= lat <= n):
                continue
            if o[1] is None:
                with open(p, encoding="utf-8") as f:
                    o[1] = make_valid(shape(json.load(f)["features"][0]["geometry"]))
                prepare(o[1])
            if contains_xy(o[1], lng, lat):
                return True
        return False


def norm(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^\w]+", " ", s).strip()


def build(city, out_dir=None):
    with open(os.path.join(PACKS, f"{city}.boundary.geojson"), encoding="utf-8") as f:
        fc = json.load(f)
    foot = make_valid(shape(fc["features"][0]["geometry"]))
    prepare(foot)  # T-425
    w, s, e, n = foot.bounds
    py = Gaps.GAP_M * 1.2 / 110540  # read Gaps' reach past the outline too
    px = py / max(0.1, math.cos(math.radians(max(abs(s), abs(n)))))
    bbox = (round(s - py, 4), round(w - px, 4), round(n + py, 4), round(e + px, 4))
    gaps = Gaps(city, foot)
    import build_city
    cfg = build_city.CITIES.get(city, {})
    extract = None
    if not os.path.exists(os.path.join(RAW, f"osm_places_{city}.json")):
        extract = cfg.get("extract")
    latin = cfg.get("latin_names", False)
    if callable(latin):  # a planned city decides from its units' OSM names (osm_units.latin_names, T-537)
        latin = latin()
    ja = __import__("japan") if cfg.get("kana_romaji") else None
    n_latin, name_src = 0, {}
    data = from_extract(extract, bbox) if extract else fetch(city, bbox)
    source = f"OpenStreetMap contributors, ODbL (Geofabrik {__import__('osm_features').label(extract)})" if extract else "OpenStreetMap contributors, ODbL (Overpass)"
    rows, n_gap = [], 0
    # T-3623: an area holding a node of its own name is that node's place (Cairo's governorate-sized
    # place=city area has its box centre in the desert 37 km from the city's node).
    node_at = {}
    for el in data.get("elements", []):
        name = (el.get("tags", {}).get("name") or "").strip()
        if el["type"] == "node" and el.get("tags", {}).get("place") in KINDS and name:
            node_at.setdefault(norm(name), []).append((el["lon"], el["lat"]))
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
            if el["type"] != "node" or not gaps.holds(lng, lat):
                continue
            n_gap += 1
        fold = None
        if el.get("geom") is not None:
            fold = next((p for p in node_at.get(norm(name), []) if contains_xy(el["geom"], *p)), None)
        pop = 0
        m = re.match(r"^\s*(\d[\d,]*)", tags.get("population", ""))
        if m:
            pop = int(m.group(1).replace(",", ""))
        others = []
        if latin:  # T-291: a Latin name to show, the local one kept for search
            src = next((k for k in LATIN_NAMES if (tags.get(k) or "").strip()), None)
            shown = tags[src].strip() if src else ""
            if not shown and ja:  # T-420: romaji from the kana reading; T-426: else from a reading dictionary
                src = next((k for k in KANA_NAMES if (tags.get(k) or "").strip()), None)
                shown = (ja.kana_romaji(tags[src]) or "") if src else ""
                src = "kana" if shown else None
                if not shown:
                    shown = ja.kanji_romaji(name) or ""
                    src = "dictionary" if shown else None
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
                     "node": el["type"] == "node", "fold": fold})
    # One row per place: OSM often has a label node and a boundary for the same neighbourhood.
    # Same name within DEDUP_M, or an area holding its node: keep the node, then the bigger kind;
    # merge other names, population. An area whose node another pack holds is dropped; one whose
    # node is in no pack stays, the place's only row. A folded area still twins later same-name areas
    # near it (OSM's second outline of a village), so they go where it went.
    rows.sort(key=lambda r: (not r["node"], r["fold"] is None, KINDS.index(r["kind"])))
    kept, by_name, at, n_fold = [], {}, {}, 0
    for r in rows:
        k = norm(r["name"])
        twin = None
        for o in by_name.get(k, []):
            dx = (o["lng"] - r["lng"]) * 111320 * 0.76
            dy = (o["lat"] - r["lat"]) * 110540
            if (dx * dx + dy * dy) ** 0.5 < DEDUP_M:
                twin = o.get("to", o)
                break
        if twin is DROP:
            continue
        if r["node"]:
            at[(r["lng"], r["lat"])] = twin or r
        elif r["fold"] and not twin:
            twin = at.get(r["fold"])
            if not twin and gaps.elsewhere(*r["fold"]):
                twin = DROP
            if twin:
                n_fold += 1
                by_name.setdefault(k, []).append({"lng": r["lng"], "lat": r["lat"], "to": twin})
            if twin is DROP:
                continue
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
    path = os.path.join(out_dir or PACKS, f"{city}.places.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    counts = {k: 0 for k in KINDS}
    for r in kept:
        counts[r["kind"]] += 1
    print(f"{city}: {len(places)} places ({len(rows)} before merging, {n_gap} in a gap of the outline, "
          f"{n_fold} areas folded into their node), {os.path.getsize(path) // 1024} KB: "
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
