"""Name each ascent-region after its highest named peak, from OpenStreetMap.

OSM `natural=peak` nodes carry name + ele + coordinates globally. We fetch every
named peak in the tile's bbox (Overpass API, cached to disk), drop each onto the
region its coordinates fall in, and name the region after the highest such peak.
"""
import os, re, json, time
import urllib.request, urllib.parse, urllib.error
import numpy as np

# public Overpass mirrors; rotate on rate-limit
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
UA = "ascentshed-map/1.0 (personal research project)"


def _query(s, w, n, e, timeout=180, retries=6):
    q = (f'[out:json][timeout:{timeout}];'
         f'node["natural"="peak"]["name"]({s},{w},{n},{e});out;')
    data = urllib.parse.urlencode({"data": q}).encode()
    last = None
    for attempt in range(retries):
        url = OVERPASS[attempt % len(OVERPASS)]
        req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=timeout + 30) as r:
                return json.load(r)["elements"]
        except urllib.error.HTTPError as ex:
            last = ex
            if ex.code in (429, 504, 503):            # rate-limited / busy: back off
                time.sleep(5 * (attempt + 1))
                continue
            raise
        except Exception as ex:
            last = ex
            time.sleep(3 * (attempt + 1))
    raise last


def fetch_peaks(bounds, cache, step=5.0):
    """bounds = rasterio bounds (left,bottom,right,top). Returns list of peak dicts
    {name, name_en, lat, lon, ele}. Tiles the bbox into `step`-degree chunks so no
    single Overpass query is huge; caches the merged result to `cache`."""
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    w, s, e, n = bounds.left, bounds.bottom, bounds.right, bounds.top
    out, seen = [], set()
    ys = np.arange(s, n, step)
    xs = np.arange(w, e, step)
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + step, n), min(x0 + step, e)
            try:
                els = _query(y0, x0, y1, x1)
            except Exception as ex:
                print(f"  overpass chunk ({y0:.0f},{x0:.0f}) failed: {ex}")
                continue
            for el in els:
                if el.get("id") in seen:
                    continue
                seen.add(el.get("id"))
                t = el.get("tags", {})
                out.append({"name": t.get("name"),
                            "name_en": t.get("name:en") or t.get("name:ja_rm"),
                            "ele": _parse_ele(t.get("ele")),
                            "lat": el["lat"], "lon": el["lon"]})
            print(f"  chunk ({y0:.0f}..{y1:.0f}, {x0:.0f}..{x1:.0f}): "
                  f"{len(els)} peaks (total {len(out)})", flush=True)
            time.sleep(2)                              # be polite to the API
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    return out


def _parse_ele(s):
    if not s:
        return None
    m = re.match(r"\s*(-?\d+(?:\.\d+)?)", str(s).replace(",", ""))
    return float(m.group(1)) if m else None


def name_regions(peaks, terr, transform, ds, z):
    """Assign each region the highest named peak whose coordinates fall in it.
    Returns {region_id: {name, name_en, lat, lon, ele, row, col}}."""
    inv = ~transform                                  # (lon,lat) -> (col,row) in src px
    H, W = terr.shape
    best = {}
    for p in peaks:
        col, row = inv * (p["lon"], p["lat"])
        i, j = int(row // ds), int(col // ds)
        if not (0 <= i < H and 0 <= j < W):
            continue
        r = int(terr[i, j])
        if r < 0:
            continue
        ele = p["ele"] if p["ele"] is not None else float(z[i, j])
        if r not in best or ele > best[r]["ele"]:
            best[r] = {"name": p["name"], "name_en": p["name_en"],
                       "lat": p["lat"], "lon": p["lon"], "ele": ele,
                       "row": i, "col": j}
    return best
